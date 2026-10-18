"""Staged CPU/Metal wall time, best of five interleaved fresh processes."""

import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from workloads import _npmpc as problem
from workloads.npmpc import function

import aotax
from aotax.sparse import sparse_hessian


def cases():
  yield "tiny", lambda x: jnp.sin(x) * x + 2, [np.ones(1, np.float32)]
  for n, batches in [(12, [1, 2, 4, 8, 16, 32, 40, 48, 64, 96, 128]), (50, [1, 2, 4, 8, 10, 12, 16, 32, 64, 128]), (200, [])]:
    inputs = [x.astype(np.float32) for x in problem.sample_inputs(n)]
    fn = sparse_hessian(function(n), *inputs, method="asdex").values
    yield f"N{n}_single", fn, inputs
    for batch in batches:
      mapped = jax.vmap(fn, in_axes=(0, None, 0))
      args = [np.tile(inputs[0], (batch, 1)), inputs[1], np.tile(inputs[2], (batch, 1))]
      yield f"N{n}_B{batch}", mapped, args


def build(directory, fn, args):
  directory.mkdir(parents=True, exist_ok=True)
  reference = np.asarray(jax.jit(fn)(*args))
  libraries = [aotax.compile(fn, *args, out=directory / device, name=device, device=device) for device in ("cpu", "metal")]
  for i, x in enumerate([*args, reference]):
    x.tofile(directory / f"{i}.bin")
  sizes = [x.size for x in [*args, reference]]
  pointers = ", ".join(f"p[{i}]" for i in range(len(sizes)))
  setup = "\n".join(f"float* p{i} = read_array(argv[1], {i}, {size});" for i, size in enumerate(sizes))
  source = directory / "bench.c"
  source.write_text(f"""#include "cpu/cpu.h"
#include "metal/metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) {{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }}
static float* read_array(const char* dir, int i, size_t count) {{
  char path[4096]; snprintf(path, sizeof(path), "%s/%d.bin", dir, i);
  FILE* f = fopen(path, "rb"); float* p = malloc(count * sizeof(float));
  if (!f || !p || fread(p, sizeof(float), count, f) != count) exit(2); fclose(f); return p;
}}
int main(int argc, char** argv) {{
  {setup}
  float* p[] = {{{", ".join(f"p{i}" for i in range(len(sizes) - 1))}, calloc({reference.size}, sizeof(float))}};
  cpu_context* cpu; metal_context* metal;
  if (cpu_create(&cpu) || metal_create(&metal)) return 1;
  double times[2] = {{0}}, error[2] = {{0}};
  for (int pass = 0; pass < 2; ++pass) {{
    int device = (pass + atoi(argv[2])) % 2;
    for (int warm = 0; warm < 30; ++warm)
      if (device ? metal_call(metal, {pointers}) : cpu_call(cpu, {pointers})) return 1;
    for (size_t i = 0; i < {reference.size}; ++i) {{
      float got = p[{len(args)}][i], expected = p{len(args)}[i];
      if (!isfinite(got) || fabs(got - expected) > 3e-4 * (1 + fabs(expected))) {{
        fprintf(stderr, "entry %zu: %.9g != %.9g\\n", i, got, expected); return 3;
      }}
      error[device] = fmax(error[device], fabs(got - expected));
    }}
    double start = now();
    for (int i = 0; i < 10; ++i) if (device ? metal_call(metal, {pointers}) : cpu_call(cpu, {pointers})) return 1;
    double us = (now() - start) * 1e5;
    int reps = device ? atoi(argv[3]) : (int)fmax(20, fmin(100000, 20000 / us));
    start = now();
    for (int i = 0; i < reps; ++i) if (device ? metal_call(metal, {pointers}) : cpu_call(cpu, {pointers})) return 1;
    times[device] = (now() - start) * 1e6 / reps;
  }}
  cpu_destroy(cpu); metal_destroy(metal);
  printf("{{\\"cpu\\":%.6f,\\"metal\\":%.6f,\\"cpu_error\\":%.9g,\\"metal_error\\":%.9g}}\\n", times[0], times[1], error[0], error[1]);
  return 0;
}}
""")
  subprocess.run(
    [
      "cc",
      "-O2",
      source,
      *(lib.archive for lib in libraries),
      "-lm",
      "-framework",
      "Accelerate",
      "-framework",
      "Metal",
      "-framework",
      "Foundation",
      "-o",
      directory / "bench",
    ],
    check=True,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("build/metal"))
  parser.add_argument("--cases", nargs="*")
  parser.add_argument("--passes", type=int, default=5)
  parser.add_argument("--reps", type=int, default=100)
  parser.add_argument("--measure-only", action="store_true")
  parser.add_argument("--build-only", action="store_true")
  options = parser.parse_args()
  selected = []
  for name, fn, args in cases():
    if options.cases and name not in options.cases:
      continue
    selected.append(name)
    if not options.measure_only:
      print(f"Building {name}", flush=True)
      build(options.out / name, fn, args)
  if options.build_only:
    return
  results = {name: [] for name in selected}
  for repeat in range(options.passes):
    for name in selected:
      directory = options.out / name
      run = subprocess.run([directory / "bench", directory, str(repeat), str(options.reps)], check=True, capture_output=True, text=True)
      record = json.loads(run.stdout)
      results[name].append(record)
      print(f"{repeat + 1} {name}: CPU {record['cpu']:.1f}, Metal {record['metal']:.1f} us", flush=True)
  (options.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
  print("| case | CPU us | Metal us | CPU/Metal |\n|---|---:|---:|---:|")
  for name, rounds in results.items():
    cpu, metal = (min(r[device] for r in rounds) for device in ("cpu", "metal"))
    print(f"| {name} | {cpu:.2f} | {metal:.1f} | {cpu / metal:.2f} |")


if __name__ == "__main__":
  main()
