"""Compile a JAX function ahead of time, through IREE, into a static C library and its header."""

import ctypes
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from iree.compiler.tools import binaries
from jax.core import ShapedArray

from aotax.blas import contractions
from aotax.custom_calls import rewrite_custom_calls
from aotax.exp import expand_exp
from aotax.host import binding_writes, host

__all__ = ["Library", "compile"]

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

  def load(self) -> Callable[..., Any]:
    """The library as a Python function of NumPy arrays, through a shared library linked from the archive."""
    return _Loaded(self)


def compile(fn: Callable[..., Any], *args: Any, out: str | Path, name: str | None = None) -> Library:
  """Compile `fn` at the shapes and dtypes of `args`, arrays or `jax.ShapeDtypeStruct`s, into `out`.

  `out` receives `{name}.h` and `lib{name}.a`, and `iree/` beneath it the intermediate files: the
  StableHLO, IREE's hardware abstraction layer (HAL) program, dispatch object and generated host.
  """
  name = name or getattr(fn, "__name__", "")
  if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
    raise ValueError(f"{name!r} is not a C identifier; pass name=")
  out = Path(out)
  work = out / "iree"
  work.mkdir(parents=True, exist_ok=True)

  # jax.export names the module after the function, `jit_{name}`, and IREE every C symbol after the module.
  def wrapped(*a: Any) -> Any:
    return fn(*a)

  wrapped.__name__ = name
  exported = jax.export.export(jax.jit(wrapped))(*args)
  for aval in (*exported.in_avals, *exported.out_avals):
    if aval.dtype not in DTYPES:
      raise TypeError(f"{name}: dtype {aval.dtype} is not supported; use one of {', '.join(map(str, DTYPES))}")
  library = Library(name, out / f"{name}.h", out / f"lib{name}.a", exported.in_avals, exported.out_avals, exported.in_tree, exported.out_tree)

  text, custom_calls = rewrite_custom_calls(expand_exp(exported.mlir_module()))
  (work / "stablehlo.mlir").write_text(text)
  sources = work / "sources"
  sources.mkdir(exist_ok=True)
  for path in sources.glob("*.mlir"):
    path.unlink()
  for path in (work / "library.o", work / "library.h"):
    path.unlink(missing_ok=True)
  subprocess.run(
    [
      binaries.find_tool("iree-compile"),
      work / "stablehlo.mlir",
      *IREE_FLAGS,
      f"--iree-hal-dump-executable-sources-to={sources}",
      f"--iree-llvmcpu-static-library-output-path={work / 'library.o'}",
      "-o",
      work / "hal.mlir",
    ],
    check=True,
  )
  library.header.write_text(_header(library))
  queries = re.findall(r"(\w+_library_query)\(", (work / "library.h").read_text()) if (work / "library.h").exists() else []
  blas = contractions(sources) if sys.platform == "darwin" else {}
  (work / "host.c").write_text(
    host(
      library,
      (work / "hal.mlir").read_text(),
      queries,
      _signature(library),
      blas,
      binding_writes(sources),
      exported.module_kept_var_idx,
      custom_calls,
    )
  )
  cc = os.environ.get("CC", "cc")
  subprocess.run(
    [
      cc,
      "-O2",
      "-fPIC",
      "-std=c11",
      "-I",
      out,
      "-c",
      work / "host.c",
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


def _header(lib: Library) -> str:
  shapes = [f"//   in{i}: {_c_type(a)}{list(a.shape)}" for i, a in enumerate(lib.in_avals)]
  shapes += [f"//   out{i}: {_c_type(a)}{list(a.shape)}" for i, a in enumerate(lib.out_avals)]
  return f"""\
// {lib.name}, compiled from a JAX function by aotax. Link lib{lib.name}.a, then {" ".join(SYSTEM_LIBS)}.
// Arrays are contiguous and row-major:
{chr(10).join(shapes)}
// create and call return 0 on success; on failure they print an error to stderr and return 1.
// Pointers need no special alignment. Read-only bindings aligned to 64 bytes avoid staging copies.
// Inputs are copied before execution if the compiled program writes their bindings.
// Inputs may alias outputs; outputs may overlap and are written in the order listed.
// A context is not thread-safe, but separate contexts may be used concurrently.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

typedef struct {lib.name}_context {lib.name}_context;

int {lib.name}_create({lib.name}_context** context);
{_signature(lib)};
void {lib.name}_destroy({lib.name}_context* context);

#ifdef __cplusplus
}}
#endif
"""


class _Loaded:
  def __init__(self, lib: Library):
    self._lib = lib
    prefix = "_" if sys.platform == "darwin" else ""
    keep = [f"-Wl,-u,{prefix}{lib.name}_{fn}" for fn in ("create", "call", "destroy")]
    with tempfile.NamedTemporaryFile(dir=lib.archive.parent, suffix=".dylib" if sys.platform == "darwin" else ".so") as shared:
      subprocess.run([os.environ.get("CC", "cc"), "-shared", "-o", shared.name, *keep, lib.archive, *SYSTEM_LIBS], check=True)
      self._dll = ctypes.CDLL(shared.name)
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
