"""Compile a JAX function ahead of time, through IREE, into a static C library and its header."""

import ctypes
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from iree.compiler.tools import binaries
from jax.core import ShapedArray

from aotax.runtime import LIBRARIES, runtime

__all__ = ["Library", "compile"]

# The dtypes a library can take and return: their C type and IREE's element type.
DTYPES = {
  np.dtype(np.float32): ("float", "IREE_HAL_ELEMENT_TYPE_FLOAT_32"),
  np.dtype(np.float64): ("double", "IREE_HAL_ELEMENT_TYPE_FLOAT_64"),
  np.dtype(np.int32): ("int32_t", "IREE_HAL_ELEMENT_TYPE_INT_32"),
  np.dtype(np.int64): ("int64_t", "IREE_HAL_ELEMENT_TYPE_INT_64"),
}

IREE_FLAGS = (
  "--iree-input-type=stablehlo",
  # IREE otherwise rewrites every f64 to f32 and every i64 to i32, silently.
  "--iree-input-demote-f64-to-f32=false",
  "--iree-input-demote-i64-to-i32=false",
  "--iree-hal-target-device=local",
  "--iree-hal-local-target-device-backends=llvm-cpu",
  "--iree-llvmcpu-target-cpu=host",
  # The dispatches as a plain object whose sin, cos and exp resolve against libm, instead of IREE's
  # self-contained ELF, which has no f64 versions of them.
  "--iree-llvmcpu-link-embedded=false",
  "--iree-llvmcpu-link-static",
  # The program that drives the dispatches as C rather than as bytecode for the runtime to interpret.
  "--output-format=vm-c",
)

# What a program linking the archive needs besides it.
SYSTEM_LIBS = ("-lm",) if sys.platform == "darwin" else ("-lm", "-lpthread", "-ldl")


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
  StableHLO, IREE's VM program as C, the dispatches' object and the host that runs them.
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

  (work / "stablehlo.mlir").write_text(exported.mlir_module())
  subprocess.run(
    [
      binaries.find_tool("iree-compile"),
      work / "stablehlo.mlir",
      *IREE_FLAGS,
      f"--iree-llvmcpu-static-library-output-path={work / 'library.o'}",
      "-o",
      work / "module.h",
    ],
    check=True,
  )
  library.header.write_text(_header(library))
  # `{module}_linked`, or the executable's own name when the module has only one.
  query = re.search(r"(\w+_library_query)\(", (work / "library.h").read_text())
  assert query, "library.h declares no library query"
  (work / "host.c").write_text(_host(library, query[1]))
  prefix = runtime()
  cc = os.environ.get("CC", "cc")
  subprocess.run(
    [
      cc,
      "-O2",
      "-fPIC",
      "-isystem",
      prefix / "include",
      "-I",
      work,
      "-I",
      out,
      # The allocator IREE's CMake package would define for its users.
      "-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl",
      "-c",
      work / "host.c",
      "-o",
      work / "host.o",
    ],
    check=True,
  )
  _merge(library.archive, [work / "host.o", work / "library.o"], [prefix / "lib" / f"lib{lib}.a" for lib in LIBRARIES])
  return library


def _merge(archive: Path, objects: list[Path], archives: list[Path]) -> None:
  """One archive holding the objects and every member of the archives, so a program links it alone, in any order."""
  archive.unlink(missing_ok=True)
  if sys.platform == "darwin":
    subprocess.run(["libtool", "-static", "-no_warning_for_no_symbols", "-o", archive, *objects, *archives], check=True, capture_output=True)
    return
  script = [f"CREATE {archive}", *(f"ADDMOD {o}" for o in objects), *(f"ADDLIB {a}" for a in archives), "SAVE", "END", ""]
  subprocess.run(["ar", "-M"], input="\n".join(script), text=True, check=True)


def _c_type(aval: ShapedArray) -> str:
  return DTYPES[np.dtype(aval.dtype)][0]


def _signature(lib: Library) -> str:
  inputs = [f"const {_c_type(a)}* in{i}" for i, a in enumerate(lib.in_avals)]
  outputs = [f"{_c_type(a)}* out{i}" for i, a in enumerate(lib.out_avals)]
  return f"int {lib.name}_call({lib.name}_context* context, {', '.join(inputs + outputs)})"


def _header(lib: Library) -> str:
  shapes = [f"//   in{i}: {_c_type(a)}{list(a.shape)}" for i, a in enumerate(lib.in_avals)]
  shapes += [f"//   out{i}: {_c_type(a)}{list(a.shape)}" for i, a in enumerate(lib.out_avals)]
  return f"""\
// {lib.name}, compiled from a JAX function by aotax. Link lib{lib.name}.a, then {" ".join(SYSTEM_LIBS)}.
// Arrays are contiguous and row-major:
{chr(10).join(shapes)}
// Each function returns 0 on success; on failure it prints IREE's error to stderr and returns 1.
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


def _host(lib: Library, query: str) -> str:
  n, module = lib.name, f"jit_{lib.name}"
  shapes, inputs = [], []
  for i, a in enumerate(lib.in_avals):
    if a.shape:
      shapes.append(f"static const iree_hal_dim_t in{i}_shape[] = {{{', '.join(map(str, a.shape))}}};")
    shape = f"in{i}_shape" if a.shape else "NULL"
    element = DTYPES[np.dtype(a.dtype)][1]
    inputs.append(
      f"  if (iree_status_is_ok(status)) status = allocate_input(c, {i}, {len(a.shape)}, {shape}, {element}, {a.size * a.dtype.itemsize});"
    )
  writes = [
    f"    status = iree_hal_buffer_map_write(iree_hal_buffer_view_buffer(c->inputs[{i}]), 0, in{i}, {a.size * a.dtype.itemsize});"
    for i, a in enumerate(lib.in_avals)
  ]
  reads = [f"    status = read_output(c, out{i}, {a.size * a.dtype.itemsize});" for i, a in enumerate(lib.out_avals)]
  # Chained so that the first failure skips everything after it.
  writes = "\n".join(f"  if (iree_status_is_ok(status)) {{\n{w[2:]}\n  }}" for w in writes)
  reads = "\n".join(f"  if (iree_status_is_ok(status)) {{\n{r[2:]}\n  }}" for r in reads)
  return f"""\
// The host aotax generates for {n}: it runs IREE's VM program for the function, module.h, whose
// dispatches library.o holds, on a single-threaded local-sync device.
#include <stdio.h>
#include <stdlib.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/runtime/api.h"
#include "library.h"
#define EMITC_IMPLEMENTATION
#include "module.h"
#include "{n}.h"

#define N_INPUTS {len(lib.in_avals)}

struct {n}_context {{
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
  iree_runtime_call_t call;
  int call_initialized;
  // Allocated once; each call copies the caller's inputs into them.
  iree_hal_buffer_view_t* inputs[N_INPUTS];
}};

{chr(10).join(shapes)}

static int report(iree_status_t status) {{
  if (iree_status_is_ok(status)) return 0;
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return 1;
}}

static iree_status_t allocate_input({n}_context* c, int i, iree_host_size_t rank, const iree_hal_dim_t* shape, iree_hal_element_type_t element,
                                    iree_device_size_t bytes) {{
  iree_hal_buffer_params_t params = {{.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
                                     .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING}};
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(c->device), params, bytes, &buffer);
  if (iree_status_is_ok(status)) {{
    status = iree_hal_buffer_view_create(buffer, rank, shape, element, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(), &c->inputs[i]);
  }}
  iree_hal_buffer_release(buffer);
  return status;
}}

// The device, with the dispatches looked up in the linked library, buffers from the heap and the
// asynchronous I/O pool the runtime requires every device to have; then a session with the module.
static iree_status_t create({n}_context* c) {{
  iree_allocator_t host = iree_allocator_system();
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  iree_hal_executable_loader_t* loader = NULL;
  iree_hal_allocator_t* allocator = NULL;
  iree_async_proactor_pool_t* pool = NULL;
  iree_vm_module_t* module = NULL;
  const iree_hal_executable_library_query_fn_t libraries[] = {{{query}}};

  iree_status_t status = iree_runtime_instance_create(&instance_options, host, &instance);
  if (iree_status_is_ok(status)) status = iree_hal_static_library_loader_create(1, libraries, iree_hal_executable_import_provider_null(), host, &loader);
  if (iree_status_is_ok(status)) status = iree_hal_allocator_create_heap(iree_make_cstring_view("local"), host, host, &allocator);
  if (iree_status_is_ok(status)) status = iree_async_proactor_pool_create(iree_numa_node_count(), NULL, iree_async_proactor_pool_options_default(), host, &pool);
  if (iree_status_is_ok(status)) {{
    iree_hal_device_create_params_t create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = pool;
    iree_hal_sync_device_params_t params;
    iree_hal_sync_device_params_initialize(&params);
    status = iree_hal_sync_device_create(iree_make_cstring_view("local-sync"), &params, &create_params, 1, &loader, allocator, host, &c->device);
  }}
  if (iree_status_is_ok(status)) {{
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(instance, &session_options, c->device, host, &c->session);
  }}
  if (iree_status_is_ok(status)) status = {module}_create(iree_runtime_instance_vm_instance(instance), host, &module);
  if (iree_status_is_ok(status)) status = iree_runtime_session_append_module(c->session, module);
  if (iree_status_is_ok(status)) status = iree_runtime_call_initialize_by_name(c->session, iree_make_cstring_view("{module}.main"), &c->call);
  if (iree_status_is_ok(status)) c->call_initialized = 1;
{chr(10).join(inputs)}
  iree_vm_module_release(module);
  iree_async_proactor_pool_release(pool);
  iree_hal_allocator_release(allocator);
  iree_hal_executable_loader_release(loader);
  iree_runtime_instance_release(instance);
  return status;
}}

int {n}_create({n}_context** context) {{
  {n}_context* c = calloc(1, sizeof(*c));
  if (!c) return report(iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "allocating the context"));
  int failed = report(create(c));
  if (failed) {{
    {n}_destroy(c);
    c = NULL;
  }}
  *context = c;
  return failed;
}}

void {n}_destroy({n}_context* c) {{
  if (!c) return;
  for (int i = 0; i < N_INPUTS; ++i) iree_hal_buffer_view_release(c->inputs[i]);
  if (c->call_initialized) iree_runtime_call_deinitialize(&c->call);
  iree_runtime_session_release(c->session);
  iree_hal_device_release(c->device);
  free(c);
}}

// The runtime allocates every result, so each is copied to caller memory and released.
static iree_status_t read_output({n}_context* c, void* out, iree_device_size_t bytes) {{
  iree_hal_buffer_view_t* result = NULL;
  iree_status_t status = iree_runtime_call_outputs_pop_front_buffer_view(&c->call, &result);
  if (iree_status_is_ok(status)) status = iree_hal_buffer_map_read(iree_hal_buffer_view_buffer(result), 0, out, bytes);
  iree_hal_buffer_view_release(result);
  return status;
}}

{_signature(lib)} {{
  {n}_context* c = context;
  iree_runtime_call_reset(&c->call);
  iree_status_t status = iree_ok_status();
{writes}
  for (int i = 0; i < N_INPUTS && iree_status_is_ok(status); ++i) status = iree_runtime_call_inputs_push_back_buffer_view(&c->call, c->inputs[i]);
  if (iree_status_is_ok(status)) status = iree_runtime_call_invoke(&c->call, 0);
{reads}
  return report(status);
}}
"""


class _Loaded:
  def __init__(self, lib: Library):
    self._lib = lib
    shared = lib.archive.with_suffix(".dylib" if sys.platform == "darwin" else ".so")
    prefix = "_" if sys.platform == "darwin" else ""
    keep = [f"-Wl,-u,{prefix}{lib.name}_{fn}" for fn in ("create", "call", "destroy")]
    subprocess.run([os.environ.get("CC", "cc"), "-shared", "-o", shared, *keep, lib.archive, *SYSTEM_LIBS], check=True)
    self._dll = ctypes.CDLL(str(shared))
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
