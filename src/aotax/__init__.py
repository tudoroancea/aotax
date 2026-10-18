"""Compile a JAX function ahead of time, through IREE, into a static C library and its header."""

import ctypes
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from iree.compiler.tools import binaries
from jax.core import ShapedArray

from aotax.blas import contractions
from aotax.cpu import fill_small_tiles
from aotax.custom_calls import FFI_HEADER, Foreign, rewrite_custom_calls
from aotax.host import binding_writes, host
from aotax.math import expand_math

__all__ = ["Foreign", "Library", "compile"]

# The dtypes a library can take and return, and their C types.
DTYPES = {
  np.dtype(np.float32): "float",
  np.dtype(np.float64): "double",
  np.dtype(np.int32): "int32_t",
  np.dtype(np.int64): "int64_t",
}

IREE_FLAGS = (
  "--iree-input-type=stablehlo",
  # IREE otherwise rewrites every f64 to f32 and every i64 to i32, silently.
  "--iree-input-demote-f64-to-f32=false",
  "--iree-input-demote-i64-to-i32=false",
  "--iree-hal-target-device=local",
  "--iree-hal-local-target-device-backends=llvm-cpu",
  "--iree-llvmcpu-target-cpu=host",
  # The host runs on one thread, so distributing dispatches only adds overhead.
  "--iree-llvmcpu-disable-distribution",
  # Unroll short dispatch loops to reduce branch overhead.
  "--iree-llvmcpu-loop-unrolling",
  # The dispatches as a plain object whose transcendental calls resolve against libm, instead of IREE's
  # self-contained ELF, which has no f64 versions of them.
  "--iree-llvmcpu-link-embedded=false",
  "--iree-llvmcpu-link-static",
  "--compile-to=hal",
  "--iree-hal-memoization=false",
)

# What a program linking the archive needs besides it.
SYSTEM_LIBS = ("-lm", "-framework", "Accelerate") if sys.platform == "darwin" else ("-lm",)


@dataclass(frozen=True)
class Library:
  """A compiled function: `header` declares `{name}_create`, `{name}_call` and `{name}_destroy`, which `archive` defines.

  The C function takes the flattened inputs, then the flattened outputs, each as a pointer to
  contiguous row-major memory. `in_avals` and `out_avals` give their shapes and dtypes, and
  `in_tree` and `out_tree` how they nest in the JAX function's arguments and result.
  """

  name: str
  header: Path
  archive: Path
  in_avals: tuple[ShapedArray, ...]
  out_avals: tuple[ShapedArray, ...]
  in_tree: Any
  out_tree: Any
  device: str = "cpu"

  @property
  def system_libs(self) -> tuple[str, ...]:
    return ("-framework", "Metal", "-framework", "Foundation") if self.device == "metal" else SYSTEM_LIBS

  def load(self, *, extra: Sequence[str | Path] = ()) -> Callable[..., Any]:
    """Load as a Python function, linking `extra` objects, libraries or flags after the archive."""
    return _Loaded(self, extra=extra)


def compile(
  fn: Callable[..., Any],
  *args: Any,
  out: str | Path,
  name: str | None = None,
  foreign: Mapping[str, str | Foreign] | None = None,
  device: str = "cpu",
) -> Library:
  """Compile `fn` at the shapes and dtypes of `args`, arrays or `jax.ShapeDtypeStruct`s, into `out`.

  `out` receives `{name}.h` and `lib{name}.a`, and `iree/` beneath it the intermediate files: the
  StableHLO, IREE's hardware abstraction layer (HAL) program, dispatch object and generated host.
  """
  if device not in {"cpu", "metal"}:
    raise ValueError(f"aotax: unsupported device {device!r}; use 'cpu' or 'metal'")
  if device == "metal" and sys.platform != "darwin":
    raise ValueError("aotax: Metal requires macOS 26 or later")
  if device == "metal":
    for leaf in jax.tree.leaves(args):
      dtype = np.dtype(leaf.dtype) if hasattr(leaf, "dtype") else np.asarray(leaf).dtype
      if dtype in {np.dtype(np.float64), np.dtype(np.int64)}:
        raise TypeError(f"aotax: Metal supports float32 and int32 only; got {dtype}")
  name = name or getattr(fn, "__name__", "")
  if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
    raise ValueError(f"{name!r} is not a C identifier; pass name=")
  targets = {target: Foreign(spec) if isinstance(spec, str) else spec for target, spec in (foreign or {}).items()}
  for target, spec in targets.items():
    if not isinstance(target, str) or not target or not isinstance(spec, Foreign):
      raise ValueError("aotax: foreign must map nonempty target names to C symbols or Foreign declarations")
  out = Path(out)
  work = out / "iree"
  work.mkdir(parents=True, exist_ok=True)

  # jax.export names the module after the function, `jit_{name}`, and IREE every C symbol after the module.
  def wrapped(*a: Any) -> Any:
    return fn(*a)

  wrapped.__name__ = name
  checks = [jax.export.DisabledSafetyCheck.custom_call(target) for target in targets]
  exported = jax.export.export(jax.jit(wrapped), disabled_checks=checks)(*args)
  for aval in (*exported.in_avals, *exported.out_avals):
    if aval.dtype not in DTYPES:
      raise TypeError(f"{name}: dtype {aval.dtype} is not supported; use one of {', '.join(map(str, DTYPES))}")
    if device == "metal" and aval.dtype not in {np.dtype(np.float32), np.dtype(np.int32)}:
      raise TypeError(f"{name}: Metal supports float32 and int32 only; got {aval.dtype}")
  if device == "metal":
    from aotax.metal import validate

    text, custom_calls = exported.mlir_module(), {}
    validate(text)
  else:
    text, custom_calls = rewrite_custom_calls(expand_math(exported.mlir_module()), targets)
  library = Library(name, out / f"{name}.h", out / f"lib{name}.a", exported.in_avals, exported.out_avals, exported.in_tree, exported.out_tree, device)

  (work / "stablehlo.mlir").write_text(text)
  sources = work / "sources"
  sources.mkdir(exist_ok=True)
  for path in sources.glob("*.mlir"):
    path.unlink()
  for path in (work / "library.o", work / "library.h"):
    path.unlink(missing_ok=True)
  flags = IREE_FLAGS
  if device == "metal":
    flags = (
      "--iree-input-type=stablehlo",
      "--iree-input-demote-f64-to-f32=false",
      "--iree-input-demote-i64-to-i32=false",
      "--iree-hal-target-device=metal",
      "--iree-metal-compile-to-metallib=false",
      "--compile-to=hal",
      "--iree-hal-memoization=false",
    )
  input_path = work / "stablehlo.mlir"
  if device == "cpu":
    configurations = work / "configured.mlir"
    subprocess.run(
      [
        binaries.find_tool("iree-compile"),
        input_path,
        *(flag for flag in flags if not flag.startswith("--compile-to=")),
        "--compile-to=executable-configurations",
        f"--iree-hal-dump-executable-sources-to={sources}",
        f"--iree-llvmcpu-static-library-output-path={work / 'library.o'}",
        "-o",
        configurations,
      ],
      check=True,
    )
    configurations.write_text(fill_small_tiles(configurations.read_text()))
    input_path = configurations
    flags = (*flags, "--compile-from=executable-configurations")
  subprocess.run(
    [
      binaries.find_tool("iree-compile"),
      input_path,
      *flags,
      f"--iree-hal-dump-executable-sources-to={sources}",
      *([f"--iree-llvmcpu-static-library-output-path={work / 'library.o'}"] if device == "cpu" else []),
      "-o",
      work / "hal.mlir",
    ],
    check=True,
  )
  declarations = []
  for spec in dict.fromkeys(custom_calls.values()):
    if spec.foreign is None:
      continue
    tensors = [
      *(f"in{i}: {t.dtype}{list(t.shape)}" for i, t in enumerate(spec.inputs)),
      *(f"out{i}: {t.dtype}{list(t.shape)}" for i, t in enumerate(spec.results)),
    ]
    declarations.append(f"// {spec.target!r}: {', '.join(tensors)}; scratch: {spec.scratch} bytes.\n{spec.foreign.declaration}")
  library.header.write_text(_header(library, "\n".join(declarations)))
  queries = re.findall(r"(\w+_library_query)\(", (work / "library.h").read_text()) if (work / "library.h").exists() else []
  blas = contractions(sources) if sys.platform == "darwin" and device == "cpu" else {}
  source = work / ("host.m" if device == "metal" else "host.c")
  source.write_text(
    host(
      library,
      (work / "hal.mlir").read_text(),
      queries,
      _signature(library),
      blas,
      binding_writes(sources),
      exported.module_kept_var_idx,
      custom_calls,
      *([device] if device == "metal" else []),
    )
  )
  cc = os.environ.get("CC", "cc")
  subprocess.run(
    [
      cc,
      "-O2",
      "-fPIC",
      "-std=c11",
      *(["-fobjc-arc", "-fblocks", "-mmacosx-version-min=13.0", "-Wno-unguarded-availability-new"] if device == "metal" else []),
      "-I",
      out,
      "-c",
      source,
      "-o",
      work / "host.o",
    ],
    check=True,
  )
  _merge(library.archive, [work / "host.o", *([work / "library.o"] if queries else [])])
  return library


def _merge(archive: Path, objects: list[Path]) -> None:
  """One archive holding the generated host and dispatches."""
  archive.unlink(missing_ok=True)
  if sys.platform == "darwin":
    subprocess.run(["libtool", "-static", "-no_warning_for_no_symbols", "-o", archive, *objects], check=True, capture_output=True)
    return
  script = [f"CREATE {archive}", *(f"ADDMOD {o}" for o in objects), "SAVE", "END", ""]
  subprocess.run(["ar", "-M"], input="\n".join(script), text=True, check=True)


def _c_type(aval: ShapedArray) -> str:
  return DTYPES[np.dtype(aval.dtype)]


def _signature(lib: Library) -> str:
  inputs = [f"const {_c_type(a)}* in{i}" for i, a in enumerate(lib.in_avals)]
  outputs = [f"{_c_type(a)}* out{i}" for i, a in enumerate(lib.out_avals)]
  return f"int {lib.name}_call({', '.join([f'{lib.name}_context* context', *inputs, *outputs])})"


def _header(lib: Library, foreign: str = "") -> str:
  shapes = [f"//   in{i}: {_c_type(a)}{list(a.shape)}" for i, a in enumerate(lib.in_avals)]
  shapes += [f"//   out{i}: {_c_type(a)}{list(a.shape)}" for i, a in enumerate(lib.out_avals)]
  if foreign:
    foreign = f"""
// Link implementations of these foreign symbols after lib{lib.name}.a.
// Inputs are borrowed and immutable; results are fresh row-major buffers to fill completely.
// Return 0 on success, nonzero on execution failure. Do not retain any pointers after return.
// Scratch is 64-byte aligned, reused without clearing, and NULL when no bytes were requested.
{foreign}
"""
  return f"""\
// {lib.name}, compiled from a JAX function by aotax. Link lib{lib.name}.a, then {" ".join(lib.system_libs)}.
// Arrays are contiguous and row-major:
{chr(10).join(shapes)}
// create and call return 0 on success; on failure they print an error to stderr and return 1.
// Pointers need no special alignment. {"Metal stages all inputs and outputs through shared buffers." if lib.device == "metal" else "Read-only bindings aligned to 64 bytes avoid staging copies."}
// {"Metal requires macOS 26 or later and compiles embedded MSL at create." if lib.device == "metal" else "Inputs are copied before execution if the compiled program writes their bindings."}
// Inputs may alias outputs; outputs may overlap and are written in the order listed.
// A context is not thread-safe, but separate contexts may be used concurrently.
#pragma once

#include <stdint.h>
{"#include <stddef.h>" if foreign else ""}
{FFI_HEADER if foreign else ""}

#ifdef __cplusplus
extern "C" {{
#endif

typedef struct {lib.name}_context {lib.name}_context;

int {lib.name}_create({lib.name}_context** context);
{_signature(lib)};
void {lib.name}_destroy({lib.name}_context* context);
{foreign}

#ifdef __cplusplus
}}
#endif
"""


class _Loaded:
  def __init__(self, lib: Library, *, extra: Sequence[str | Path] = ()):
    self._lib = lib
    prefix = "_" if sys.platform == "darwin" else ""
    keep = [f"-Wl,-u,{prefix}{lib.name}_{fn}" for fn in ("create", "call", "destroy")]
    with tempfile.NamedTemporaryFile(dir=lib.archive.parent, suffix=".dylib" if sys.platform == "darwin" else ".so") as shared:
      linked = subprocess.run(
        [os.environ.get("CC", "cc"), "-shared", "-o", shared.name, *keep, lib.archive, *extra, *lib.system_libs],
        check=False,
        capture_output=True,
        text=True,
      )
      if linked.returncode:
        raise RuntimeError(f"{lib.name}: linking failed; pass foreign objects or libraries with load(extra=[...])\n{linked.stderr}")
      try:
        self._dll = ctypes.CDLL(shared.name)
      except OSError as error:
        raise RuntimeError(f"{lib.name}: loading failed; pass foreign objects or libraries with load(extra=[...])\n{error}") from error
    self._call = getattr(self._dll, f"{lib.name}_call")
    self._call.argtypes = [ctypes.c_void_p] * (1 + len(lib.in_avals) + len(lib.out_avals))
    self._destroy = getattr(self._dll, f"{lib.name}_destroy")
    self._destroy.argtypes = [ctypes.c_void_p]
    self._context = ctypes.c_void_p()
    if getattr(self._dll, f"{lib.name}_create")(ctypes.byref(self._context)):
      raise RuntimeError(f"{lib.name}_create failed")

  def __call__(self, *args: Any) -> Any:
    leaves, tree = jax.tree.flatten((args, {}))
    if tree != self._lib.in_tree:
      raise TypeError(f"{self._lib.name}: arguments nest as {tree}, compiled as {self._lib.in_tree}")
    inputs = []
    for i, (leaf, aval) in enumerate(zip(leaves, self._lib.in_avals, strict=True)):
      array = np.asarray(leaf, dtype=aval.dtype, order="C")
      if array.shape != aval.shape:
        raise ValueError(f"{self._lib.name}: input {i} has shape {array.shape}, compiled for {aval.shape}")
      inputs.append(array)
    outputs = [np.empty(aval.shape, aval.dtype) for aval in self._lib.out_avals]
    if self._call(self._context, *(a.ctypes.data for a in (*inputs, *outputs))):
      raise RuntimeError(f"{self._lib.name}_call failed")
    return jax.tree.unflatten(self._lib.out_tree, outputs)

  def __del__(self) -> None:
    if getattr(self, "_context", None):
      self._destroy(self._context)
