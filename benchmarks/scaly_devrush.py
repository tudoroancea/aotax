"""Study Scaly devrush's generated Newton and dense IPM without changing the installed Scaly."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent


def prepare(source: Path, directory: Path, name: str, size: str):
  sys.path.insert(0, str(source / "src"))
  os.environ["SCALY_CACHE_DIR"] = str(directory / "cache")
  sc: Any = import_module("scaly")
  write_module = import_module("scaly.codegen").write_module

  values = np.load(directory / "inputs.npz")
  if name == "newton":
    n = values["z"].size
    a = (np.diag(2 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)) * (n + 1) ** 2

    @sc.roots.root(vars=sc.L("z", n), params=sc.L("lam", ()), name="bratu")
    def root(z, lam):
      return sc.const(a) @ z - lam * z.exp()

    function = sc.roots.solver(root, sc.roots.Newton(tol=1e-10, max_iter=20, linear="lu"), name="study")
    args = (values["z"], values["lam"])
  else:
    IPM = import_module("scaly.opt.ipm.method").IPM

    q, c = values["q"], values["c"]
    n = c.size

    @sc.opt.problem(vars=sc.L("x", n), params=sc.L("c", n), name="chain_qp")
    def qp(x, c):
      return sc.opt.ProblemSpec(minimize=0.5 * sc.dot(x, sc.const(q) @ x) + sc.dot(c, x), lb=sc.const(-np.ones(n)), ub=sc.const(np.ones(n)))

    function = sc.opt.solver(qp, IPM(backend="dense", options={"max_iter": 30, "eps_abs": 1e-9, "eps_rel": 1e-9}), name="study")
    args = (np.zeros(n), np.zeros(n), np.zeros(0), np.zeros(0), c)
  module = write_module(function, directory)
  outputs = function.concrete.outputs
  np.savez(directory / "args.npz", allow_pickle=False, **{f"a{i}": a for i, a in enumerate(args)})
  (directory / "layout.json").write_text(json.dumps({"shapes": [o.shape for o in outputs], "workspace": module.workspace_size}))
  (directory / "timer.c").write_text("""#include <time.h>
#include "study.h"
double study_time(int count, const double** args, double** results, double* workspace) {
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (int i = 0; i < count; ++i) if (study(args, results, 0, workspace, 0)) return -1;
  clock_gettime(CLOCK_MONOTONIC, &b);
  return ((b.tv_sec-a.tv_sec)*1e9 + b.tv_nsec-a.tv_nsec) / count / 1e3;
}
""")
  subprocess.run(
    [
      "cc",
      "-O3",
      "-mcpu=native",
      "-fno-math-errno",
      "-shared",
      "-I",
      str(directory),
      str(directory / "study.c"),
      str(directory / "timer.c"),
      "-lm",
      "-o",
      str(directory / "timer.dylib"),
    ],
    check=True,
  )


def measure(directory: Path):
  layout = json.loads((directory / "layout.json").read_text())
  args = list(np.load(directory / "args.npz").values())
  outputs = [np.empty(shape, np.float64) for shape in layout["shapes"]]
  workspace = np.empty(max(1, layout["workspace"]), np.float64)
  dll = ctypes.CDLL(str(directory / "timer.dylib"))
  timer = dll.study_time
  timer.restype = ctypes.c_double
  timer.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  ia = (ctypes.c_void_p * len(args))(*(a.ctypes.data for a in args))
  oa = (ctypes.c_void_p * len(outputs))(*(a.ctypes.data for a in outputs))
  timer(1, ia, oa, workspace.ctypes.data)
  pilot = timer(100, ia, oa, workspace.ctypes.data)
  count = max(100, min(1_000_000, int(200_000 / pilot)))
  us = timer(count, ia, oa, workspace.ctypes.data)
  if us < 0:
    raise RuntimeError("Scaly native call failed")
  return {"us": us, "outputs": [o.tolist() for o in outputs]}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reps", type=int, default=3)
  parser.add_argument("--newton-sizes", default="4,8,16,32,64")
  parser.add_argument("--qp-sizes", default="3x4x15,3x12x15,5x12x15")
  parser.add_argument("--prepare", nargs=4, metavar=("SOURCE", "DIRECTORY", "NAME", "SIZE"))
  parser.add_argument("--measure", type=Path)
  args = parser.parse_args()
  if args.prepare:
    source, directory, name, size = args.prepare
    prepare(Path(source), Path(directory), name, size)
    return
  if args.measure:
    print(json.dumps(measure(args.measure)))
    return
  if args.reps < 1:
    parser.error("reps must be positive")
  import harness
  import program
  from workload import ProgramWorkload, discover

  checkout = Path(os.environ.get("SCALY_DEVRUSH_CHECKOUT", "/Users/tudoroancea/dev/scaly"))
  revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "origin/devrush"], text=True).strip()
  workloads = discover()
  rows = []
  with tempfile.TemporaryDirectory(prefix="aotax-devrush-") as temporary:
    root = Path(temporary)
    archive = root / "source.tar"
    with archive.open("wb") as stream:
      subprocess.run(["git", "-C", str(checkout), "archive", revision, "src/scaly"], stdout=stream, check=True)
    with tarfile.open(archive) as tar:
      tar.extractall(root / "source", filter="data")
    for name, sizes in (("newton", args.newton_sizes.split(",")), ("chain_qp", args.qp_sizes.split(","))):
      workload = workloads[name]
      assert isinstance(workload, ProgramWorkload)
      for size in sizes:
        directory = root / f"{name}-{size}"
        directory.mkdir()
        inputs = workload.inputs(size)
        keys = ("z", "lam") if name == "newton" else ("q", "c", "g", "h")
        np.savez(directory / "inputs.npz", allow_pickle=False, **dict(zip(keys, inputs, strict=True)))
        subprocess.run([sys.executable, __file__, "--prepare", str(root / "source"), str(directory), name, size], check=True)
        binary = HERE / "build" / name / f"N{size}" / "aotax" / "benchmark"
        if not binary.exists():
          binary.parent.mkdir(parents=True, exist_ok=True)
          binary = program.build(workload, size, binary.parent)
        ref = program.reference(workload, size, inputs)
        times = {b: [] for b in ("scaly", "aotax", "xla", "jax_python")}
        stats = {}
        for rep in range(args.reps):
          result = json.loads(subprocess.check_output([sys.executable, __file__, "--measure", str(directory)], text=True))
          output = [np.asarray(o) for o in result["outputs"]]
          status_index, iteration_index = (1, 2) if name == "newton" else (4, 5)
          if output[status_index].item() != 0:
            raise ValueError(f"Scaly {name}/{size}: status {output[status_index]}")
          np.testing.assert_allclose(output[0], ref[0], rtol=1e-5, atol=2e-6)
          if name == "newton":
            np.testing.assert_equal(output[iteration_index].item(), np.asarray(ref[1]).item())
          stats = {
            "scaly_iterations": output[iteration_index].item(),
            "status": output[status_index].item(),
            "solution_error": float(np.max(np.abs(output[0] - ref[0]))),
            "scaly_residual": output[-1].item(),
          }
          times["scaly"].append(result["us"])
          times["aotax"].append(harness.time(binary, "0.2s"))
          for backend, value in program.time_jax(workload, size, 0.2).items():
            times[backend].append(value)
          print(f"{name}/{size} repetition {rep + 1}/{args.reps}", flush=True)
        rows.append({"name": name, "size": size, **{b: min(v) for b, v in times.items()}, **stats, "repetitions": times})
  result = {"revision": revision, "rows": rows}
  destination = HERE / "build" / "devrush.json"
  destination.write_text(json.dumps(result, indent=2))
  print(f"Scaly origin/devrush {revision}; microseconds per evaluation")
  print(f"{'workload/size':>24} {'Scaly':>10} {'aotax':>10} {'XLA loop':>10} {'JAX Python':>12} {'iters':>7} {'residual':>12}")
  for row in rows:
    label = f"{row['name']}/{row['size']}"
    print(
      f"{label:>24} {row['scaly']:>10.2f} {row['aotax']:>10.2f} {row['xla']:>10.2f} {row['jax_python']:>12.2f} "
      f"{row['scaly_iterations']:>7.0f} {row['scaly_residual']:>12.2e}"
    )


if __name__ == "__main__":
  main()
