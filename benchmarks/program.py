"""Build complete-program binaries and time JAX in a fresh CPU process."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import harness
import jax
import jax.numpy as jnp
import numpy as np
from workload import ProgramWorkload, discover

import aotax

TYPES = {np.dtype("float64"): ("double", 0), np.dtype("int32"): ("int32_t", 1), np.dtype("int64"): ("int64_t", 2)}


def reference(workload: ProgramWorkload, size: str, inputs: tuple[Any, ...]):
  with jax.default_device(jax.devices("cpu")[0]):
    return jax.tree.map(np.asarray, jax.jit(workload.function(size))(*inputs))


def check(workload: ProgramWorkload, size: str, got, want):
  if jax.tree.structure(got) != jax.tree.structure(want):
    raise ValueError("output pytree does not match reference")
  actual, expected = jax.tree.leaves(got), jax.tree.leaves(want)
  tolerances = workload.tolerances(size)
  if len(tolerances) != len(expected):
    raise ValueError("one tolerance pair is required per output leaf")
  for a, b, (rtol, atol) in zip(actual, expected, tolerances, strict=True):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape or b.dtype not in TYPES:
      raise ValueError(f"output shape/dtype mismatch: {a.shape}/{a.dtype}, {b.shape}/{b.dtype}")
    if not np.all(np.isfinite(b)):
      raise ValueError("reference contains nonfinite values")
    if np.issubdtype(b.dtype, np.integer):
      np.testing.assert_array_equal(a, b)
    else:
      np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def build(workload: ProgramWorkload, size: str, out: Path) -> Path:
  inputs = workload.inputs(size)
  want = reference(workload, size, inputs)
  specs = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), inputs)
  library = aotax.compile(workload.function(size), *specs, out=out, name="generated")
  loaded = library.load()
  check(workload, size, loaded(*inputs), want)
  varied = perturb(inputs, 1)
  check(workload, size, loaded(*varied), reference(workload, size, varied))
  leaves, outputs = jax.tree.leaves(inputs), jax.tree.leaves(want)
  for value in (*leaves, *outputs):
    if value.dtype not in TYPES:
      raise ValueError(f"unsupported program dtype {value.dtype}")
  data = out.parent / "inputs"
  data.mkdir(exist_ok=True)
  for prefix, values in (("input", leaves), ("output", outputs)):
    for i, value in enumerate(values):
      np.asarray(value).tofile(data / f"{prefix}{i}.bin")

  def array(ctype, name, values):
    return f"static const {ctype} {name}[] = {{{', '.join(map(str, values))}}};"

  args = [
    f"static_cast<{TYPES[v.dtype][0]}*>({group}[{i}])" for group, values in (("inputs", leaves), ("outputs", outputs)) for i, v in enumerate(values)
  ]
  layout = [f"#define NINPUTS {len(leaves)}", f"#define NOUTPUTS {len(outputs)}", f"#define PROGRAM_CALL generated_call(context, {', '.join(args)})"]
  for prefix, values in (("INPUT", leaves), ("OUTPUT", outputs)):
    layout.extend(
      [
        array("std::size_t", f"{prefix}_BYTES", [v.nbytes for v in values]),
        array("char* const", f"{prefix}_NAMES", [json.dumps(f"{prefix.lower()}{i}.bin") for i in range(len(values))]),
      ]
    )
  layout.extend(
    [
      array("std::size_t", "OUTPUT_COUNTS", [v.size for v in outputs]),
      array("int", "OUTPUT_TYPES", [TYPES[v.dtype][1] for v in outputs]),
      array("double", "RTOL", [t[0] for t in workload.tolerances(size)]),
      array("double", "ATOL", [t[1] for t in workload.tolerances(size)]),
    ]
  )
  (out / "program_layout.h").write_text("\n".join(layout) + "\n")
  binary = out / "benchmark"
  harness.run(
    [
      "clang++",
      *harness.FLAGS,
      "-std=c++17",
      "-pthread",
      "-I",
      harness.gbench() / "include",
      "-I",
      out,
      f'-DINPUTS_DIR="{data}"',
      harness.CPP / "program.cpp",
      library.archive,
      *aotax.SYSTEM_LIBS,
      harness.gbench() / "lib" / "libbenchmark.a",
      "-o",
      binary,
    ]
  )
  (out / "metadata.json").write_text(
    json.dumps(
      {
        "axes": workload.axes(size),
        "outputs": [{"shape": v.shape, "dtype": str(v.dtype)} for v in outputs],
        "integer_outputs": [v.tolist() for v in outputs if np.issubdtype(v.dtype, np.integer)],
      },
      indent=2,
    )
  )
  return binary


def perturb(inputs, iteration):
  return jax.tree.map(lambda v: v * (1 + 1e-8 * (iteration % 2)) if np.issubdtype(v.dtype, np.floating) else v, inputs)


def jax_times(workload: ProgramWorkload, size: str, seconds: float) -> dict[str, float]:
  with jax.default_device(jax.devices("cpu")[0]):
    inputs = jax.tree.map(jax.device_put, workload.inputs(size))
    fn = jax.jit(workload.function(size))
    want = jax.block_until_ready(fn(*inputs))

    @jax.jit
    def many(count, args):
      def body(i, checksum):
        result = fn(*perturb(args, i))
        return checksum + sum(jnp.sum(v.astype(jnp.float64)) for v in jax.tree.leaves(result))

      return jax.lax.fori_loop(0, count, body, jnp.float64(0))

    pilot = 32
    jax.block_until_ready(many(pilot, inputs))
    start = time.perf_counter_ns()
    jax.block_until_ready(many(pilot, inputs))
    elapsed = (time.perf_counter_ns() - start) / 1e9
    count = max(32, min(100_000, int(seconds / elapsed * pilot)))
    start = time.perf_counter_ns()
    checksum = jax.block_until_ready(many(count, inputs))
    amortized = (time.perf_counter_ns() - start) / count / 1e3
    if not np.isfinite(checksum):
      raise ValueError("nonfinite timing checksum")
    check(workload, size, fn(*inputs), want)
    count = max(10, min(1000, count))
    start = time.perf_counter_ns()
    for _ in range(count):
      jax.block_until_ready(fn(*inputs))
    python = (time.perf_counter_ns() - start) / count / 1e3
    return {"xla": amortized, "jax_python": python}


def time_jax(workload: ProgramWorkload, size: str, seconds: float) -> dict[str, float]:
  out = subprocess.run([sys.executable, __file__, workload.name, size, str(seconds)], check=True, capture_output=True, text=True)
  return json.loads(out.stdout)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("workload")
  parser.add_argument("size")
  parser.add_argument("seconds", type=float)
  args = parser.parse_args()
  workload = discover()[args.workload]
  assert isinstance(workload, ProgramWorkload)
  print(json.dumps(jax_times(workload, args.size, args.seconds)))
