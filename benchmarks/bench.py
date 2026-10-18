"""Time sparse kernels and complete programs in interleaved fresh processes."""

import argparse
import csv
import math
import shutil
from pathlib import Path

import harness
import program
from workload import ProgramWorkload, discover

BACKENDS = {"scaly": harness.lower_scaly, "aotax": harness.lower_aotax, "aotax_structured": harness.lower_structured}


def main() -> None:
  workloads = discover()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workloads", type=lambda s: s.split(","), default=list(workloads), help="workload names, comma-separated")
  parser.add_argument("--sizes", type=lambda s: s.split(","), help="override every selected workload's sizes, comma-separated")
  parser.add_argument("--reps", type=int, default=3, help="fresh processes per binary; the best is reported")
  parser.add_argument("--min-time", default="0.5s", help="Google Benchmark's minimum time per process")
  args = parser.parse_args()
  if unknown := set(args.workloads) - workloads.keys():
    parser.error(f"unknown workloads {sorted(unknown)}; choose from {list(workloads)}")
  if args.reps < 1:
    parser.error("reps must be positive")
  for name in args.workloads:
    if args.sizes is not None:
      try:
        selected = workloads[name]
        if isinstance(selected, ProgramWorkload):
          for size in args.sizes:
            selected.axes(size)
        elif any(int(size) < 1 for size in args.sizes):
          raise ValueError("sizes must be positive")
      except ValueError as error:
        parser.error(str(error))
  seconds = 0.5
  if any(isinstance(workloads[name], ProgramWorkload) for name in args.workloads):
    try:
      seconds = float(args.min_time.removesuffix("s"))
    except ValueError:
      parser.error("program --min-time must be seconds, for example 0.5s")
    if not math.isfinite(seconds) or seconds <= 0:
      parser.error("program --min-time must be positive seconds")
  build = harness.HERE / "build"
  binaries: dict[tuple[str, int | str, str], Path] = {}
  program_cases = {}
  for name in args.workloads:
    workload = workloads[name]
    for raw_size in args.sizes or workload.sizes:
      if isinstance(workload, ProgramWorkload):
        size = str(raw_size)
        workload.axes(size)
        out = build / name / f"N{size}" / "aotax"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        print(f"{name} size={size}: lowering and building aotax", flush=True)
        binaries[name, size, "aotax"] = program.build(workload, size, out)
        program_cases[name, size] = workload
        continue
      size = int(raw_size)
      directory = build / name / f"N{size}" / "inputs"
      inputs = workload.inputs(size)
      reference = harness.write_inputs(directory, workload, size, inputs)
      for backend, lower in BACKENDS.items():
        out = directory.parent / backend
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        print(f"{name} size={size}: lowering and building {backend}", flush=True)
        binaries[name, size, backend] = harness.build(lower(workload, size, out, inputs), workload, inputs, reference, directory)

  times: dict[tuple[str, int | str, str], list[float]] = {key: [] for key in binaries}
  for name, size in program_cases:
    for backend in ("xla", "jax_python"):
      times[name, size, backend] = []
  for rep in range(args.reps):
    for key, binary in binaries.items():
      times[key].append(harness.time(binary, args.min_time))
      print(f"rep {rep + 1}/{args.reps} {key[0]} size={key[1]} {key[2]}: {times[key][-1]:.2f} us", flush=True)
      if (key[0], key[1]) in program_cases:
        for backend, value in program.time_jax(program_cases[key[0], str(key[1])], str(key[1]), seconds).items():
          times[key[0], key[1], backend].append(value)

  with open(build / "results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["workload", "size", "backend", "best_us", *(f"rep{i + 1}_us" for i in range(args.reps))])
    for (name, size, backend), values in times.items():
      writer.writerow([name, size, backend, f"{min(values):.2f}", *(f"{v:.2f}" for v in values)])
  for name in args.workloads:
    print(f"\n{name}, CPU time in us, best of {args.reps} processes")
    if isinstance(workloads[name], ProgramWorkload):
      print(f"{'size':>12}{'aotax':>12}{'XLA loop':>12}{'JAX Python':>12}{'aot/XLA':>9}")
    else:
      print(f"{'size':>5}{'scaly':>12}{'aotax':>12}{'structured':>12}{'ratio':>8}")
    for size in args.sizes or workloads[name].sizes:
      if isinstance(workloads[name], ProgramWorkload):
        size = str(size)
        aot, xla, python = (min(times[name, size, b]) for b in ("aotax", "xla", "jax_python"))
        print(f"{size:>12}{aot:>12.2f}{xla:>12.2f}{python:>12.2f}{aot / xla:>8.2f}x")
        continue
      size = int(size)
      scaly, aot = min(times[name, size, "scaly"]), min(times[name, size, "aotax"])
      structured = min(times[name, size, "aotax_structured"])
      print(f"{size:>5}{scaly:>12.1f}{aot:>12.1f}{structured:>12.1f}{aot / scaly:>7.2f}x")


if __name__ == "__main__":
  main()
