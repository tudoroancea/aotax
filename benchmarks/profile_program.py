"""Count dynamic dispatches and foreign calls in an already-built program."""

import argparse
import ctypes
import json
import re

import harness
import jax
import numpy as np
from workload import ProgramWorkload, discover

import aotax


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("workload")
  parser.add_argument("size")
  args = parser.parse_args()
  workload = discover()[args.workload]
  assert isinstance(workload, ProgramWorkload)
  inputs = workload.inputs(args.size)
  out = harness.HERE / "build" / workload.name / f"N{args.size}" / "aotax"
  work = out / "iree"
  source = (work / "host.c").read_text()
  source = source.replace("static int dispatch(", "uint64_t profile_dispatches=0, profile_foreign=0;\nstatic int dispatch(")
  source = source.replace(
    "  const dispatch_attrs* a = &e->exports.attrs[ordinal];", "  ++profile_dispatches;\n  const dispatch_attrs* a = &e->exports.attrs[ordinal];"
  )
  source = re.sub(r"(static int aotax_cc\d+\([^\n]+\) \{\n)", r"\1++profile_foreign;\n", source)
  source += "\nsize_t profile_context_size(void) { return sizeof(generated_context); }\n"
  (work / "profile.c").write_text(source)
  shared = out / "profile.dylib"
  objects = [work / "library.o"] if (work / "library.o").exists() else []
  harness.run(["cc", "-shared", "-O2", "-I", out, work / "profile.c", *objects, *aotax.SYSTEM_LIBS, "-o", shared])
  dll = ctypes.CDLL(str(shared))
  context = ctypes.c_void_p()
  if dll.generated_create(ctypes.byref(context)):
    raise RuntimeError("context creation failed")
  dc = ctypes.c_uint64.in_dll(dll, "profile_dispatches")
  fc = ctypes.c_uint64.in_dll(dll, "profile_foreign")
  dc.value = fc.value = 0
  exported = jax.export.export(jax.jit(workload.function(args.size)))(*inputs)
  outputs = [np.empty(a.shape, a.dtype) for a in exported.out_avals]
  call = dll.generated_call
  call.argtypes = [ctypes.c_void_p] * (1 + len(jax.tree.leaves(inputs)) + len(outputs))
  if call(context, *(v.ctypes.data for v in (*jax.tree.leaves(inputs), *outputs))):
    raise RuntimeError("native call failed")
  dll.profile_context_size.restype = ctypes.c_size_t
  result = {
    "workload": workload.name,
    "size": args.size,
    "dispatches": dc.value,
    "foreign_calls": fc.value,
    "context_struct_bytes": dll.profile_context_size(),
    "integer_outputs": [v.tolist() for v in outputs if np.issubdtype(v.dtype, np.integer)],
  }
  dll.generated_destroy.argtypes = [ctypes.c_void_p]
  dll.generated_destroy(context)
  destination = out / "profile.json"
  destination.write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
