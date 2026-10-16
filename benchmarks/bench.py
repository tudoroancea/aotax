"""Time the npmpc Lagrangian Hessian from Scaly and from aotax at each horizon, as interleaved fresh processes.

Run with `uv run --group bench benchmarks/bench.py`. Everything lands in benchmarks/build/:
third_party/ once, then for each horizon N, inputs/ and one directory per backend holding the
generated files its binary is built from, with the harness build in bin/.
"""

import argparse
import csv
import shutil

import harness

BACKENDS = {"scaly": harness.lower_scaly, "aotax": harness.lower_aotax}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--sizes", type=lambda s: [int(n) for n in s.split(",")], default=[12, 50, 200], help="horizons, comma-separated")
  parser.add_argument("--reps", type=int, default=3, help="fresh processes per binary; the best is reported")
  parser.add_argument("--min-time", default="0.5s", help="Google Benchmark's minimum time per process")
  args = parser.parse_args()
  build = harness.HERE / "build"

  binaries = {}
  for N in args.sizes:
    inputs = harness.write_inputs(build / f"N{N}" / "inputs", N)
    for name, lower in BACKENDS.items():
      out = build / f"N{N}" / name
      shutil.rmtree(out, ignore_errors=True)
      out.mkdir(parents=True)
      print(f"N={N}: lowering and building {name}", flush=True)
      binaries[N, name] = harness.build(lower(N, out), N, inputs)

  times: dict[tuple[int, str], list[float]] = {key: [] for key in binaries}
  for rep in range(args.reps):
    for key, binary in binaries.items():
      times[key].append(harness.time(binary, args.min_time))
      print(f"rep {rep + 1}/{args.reps} N={key[0]} {key[1]}: {times[key][-1]:.2f} us", flush=True)

  with open(build / "results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "backend", "best_us", *(f"rep{i + 1}_us" for i in range(args.reps))])
    for (N, name), values in times.items():
      writer.writerow([N, name, f"{min(values):.2f}", *(f"{v:.2f}" for v in values)])
  print(f"\nLower-triangle Lagrangian Hessian, gbench CPU time in us, best of {args.reps} processes")
  print(f"{'N':>5}{'scaly':>12}{'aotax':>12}{'ratio':>8}")
  for N in args.sizes:
    scaly, aot = min(times[N, "scaly"]), min(times[N, "aotax"])
    print(f"{N:>5}{scaly:>12.1f}{aot:>12.1f}{aot / scaly:>7.2f}x")


if __name__ == "__main__":
  main()
