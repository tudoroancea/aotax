"""Time sparse workloads from Scaly and aotax in interleaved fresh processes."""

import argparse
import csv
import shutil

import harness
from workload import discover

BACKENDS = {"scaly": harness.lower_scaly, "aotax": harness.lower_aotax}


def main() -> None:
  workloads = discover()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workloads", type=lambda s: s.split(","), default=list(workloads), help="workload names, comma-separated")
  parser.add_argument("--sizes", type=lambda s: [int(n) for n in s.split(",")], help="override every selected workload's sizes, comma-separated")
  parser.add_argument("--reps", type=int, default=3, help="fresh processes per binary; the best is reported")
  parser.add_argument("--min-time", default="0.5s", help="Google Benchmark's minimum time per process")
  args = parser.parse_args()
  if unknown := set(args.workloads) - workloads.keys():
    parser.error(f"unknown workloads {sorted(unknown)}; choose from {list(workloads)}")
  if args.reps < 1 or (args.sizes is not None and any(size < 1 for size in args.sizes)):
    parser.error("reps and sizes must be positive")
  build = harness.HERE / "build"
  binaries = {}
  for name in args.workloads:
    workload = workloads[name]
    for size in args.sizes or workload.sizes:
      directory = build / name / f"N{size}" / "inputs"
      inputs = workload.inputs(size)
      reference = harness.write_inputs(directory, workload, size, inputs)
      for backend, lower in BACKENDS.items():
        out = directory.parent / backend
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        print(f"{name} size={size}: lowering and building {backend}", flush=True)
        binaries[name, size, backend] = harness.build(lower(workload, size, out, inputs), workload, inputs, reference, directory)

  times: dict[tuple[str, int, str], list[float]] = {key: [] for key in binaries}
  for rep in range(args.reps):
    for key, binary in binaries.items():
      times[key].append(harness.time(binary, args.min_time))
      print(f"rep {rep + 1}/{args.reps} {key[0]} size={key[1]} {key[2]}: {times[key][-1]:.2f} us", flush=True)

  with open(build / "results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["workload", "size", "backend", "best_us", *(f"rep{i + 1}_us" for i in range(args.reps))])
    for (name, size, backend), values in times.items():
      writer.writerow([name, size, backend, f"{min(values):.2f}", *(f"{v:.2f}" for v in values)])
  for name in args.workloads:
    print(f"\n{name}, CPU time in us, best of {args.reps} processes")
    print(f"{'size':>5}{'scaly':>12}{'aotax':>12}{'ratio':>8}")
    for size in args.sizes or workloads[name].sizes:
      scaly, aot = min(times[name, size, "scaly"]), min(times[name, size, "aotax"])
      print(f"{size:>5}{scaly:>12.1f}{aot:>12.1f}{aot / scaly:>7.2f}x")


if __name__ == "__main__":
  main()
