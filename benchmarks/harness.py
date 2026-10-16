"""Lower the npmpc Hessian through Scaly or aotax, link it behind cpp/main.cpp with Google Benchmark, and time the binary."""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cmake
import hessian
import jax
import jax.numpy as jnp
import ninja
import numpy as np
import problem
from scaly.codegen.aot import render_c_module
from scaly.codegen.toolchain import find_c_compiler, native_recipe

import aotax

HERE = Path(__file__).parent
CPP = HERE / "cpp"
THIRD_PARTY = HERE / "build" / "third_party"
# The flags Scaly's own benchmark cells compile with, and the vector math library they link. macOS has no libmvec.
DARWIN = sys.platform == "darwin"
FLAGS = ("-O3", "-mcpu=native", "-fno-math-errno") if DARWIN else ("-O3", "-march=native", "-fno-math-errno", "-fveclib=libmvec")
VECTOR_LIBM = () if DARWIN else ("-lmvec",)


def run(cmd: list, cwd: Path | None = None) -> None:
  subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


@cache
def gbench() -> Path:
  """Google Benchmark at the tag Scaly's harness pins."""
  prefix = THIRD_PARTY / "gbench"
  if not (prefix / "lib" / "libbenchmark.a").exists():
    src, build = THIRD_PARTY / "gbench-src", THIRD_PARTY / "gbench-build"
    shutil.rmtree(src, ignore_errors=True)
    run(["git", "clone", "--quiet", "--depth=1", "--branch", "v1.9.5", "https://github.com/google/benchmark.git", src])
    cmake_bin = Path(cmake.CMAKE_BIN_DIR) / "cmake"
    run(
      [cmake_bin, "-S", src, "-B", build, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={Path(ninja.BIN_DIR) / 'ninja'}", "-DCMAKE_BUILD_TYPE=Release"]
      + [f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_INSTALL_LIBDIR=lib", "-DBENCHMARK_ENABLE_TESTING=OFF", "-DBENCHMARK_ENABLE_WERROR=OFF"]
    )
    run([cmake_bin, "--build", build])
    run([cmake_bin, "--install", build])
    shutil.rmtree(build)
    shutil.rmtree(src)
  return prefix


@dataclass(frozen=True)
class Kernel:
  """One backend's generated files in `dir`: `sources` to compile with the harness flags, `libs` to link, `adapter` the file in cpp/ behind kernel.h."""

  dir: Path
  rows: np.ndarray
  cols: np.ndarray
  adapter: str
  sources: tuple[Path, ...] = ()
  libs: tuple[str | Path, ...] = ()


def lower_scaly(N: int, out: Path) -> Kernel:
  """`sc.sparse_hessian` of the Scaly Lagrangian, rendered to one C source with the vector math library Scaly selects on this host."""
  function, rows, cols = hessian.scaly_lower(N)
  compiler = find_c_compiler()
  vector_libm = native_recipe(compiler.command).vector_libm if compiler is not None else "none"
  module = render_c_module(function, header_name="npmpc_hess.h", source_name="npmpc_hess.c", vector_libm=vector_libm)
  (out / module.header_name).write_text(module.header)
  (out / module.source_name).write_text(module.source)
  return Kernel(out, rows, cols, adapter="scaly.c", sources=(out / module.source_name,))


def lower_aotax(N: int, out: Path) -> Kernel:
  """asdex's sparse Hessian of the JAX Lagrangian, compiled by aotax into `npmpc_hess.h` and `libnpmpc_hess.a`."""
  function, rows, cols = hessian.asdex_lower(N)
  specs = [jax.ShapeDtypeStruct((size,), jnp.float64) for size in (problem.n_z(N), problem.N_P, problem.n_g(N))]
  library = aotax.compile(function, *specs, out=out, name="npmpc_hess")
  return Kernel(out, rows, cols, adapter="aotax.c", libs=(library.archive, *aotax.SYSTEM_LIBS))


def write_inputs(directory: Path, N: int) -> Path:
  """The sample inputs and the dense reference Hessian, as raw doubles the harness reads at startup."""
  directory.mkdir(parents=True, exist_ok=True)
  z, p, lam_g = problem.sample_inputs(N)
  for name, values in (("z", z), ("p", p), ("lam_g", lam_g), ("hessian", problem.dense_hessian(N, z, p, lam_g))):
    np.asarray(values, dtype=np.float64).tofile(directory / f"{name}.bin")
  return directory


def build(kernel: Kernel, N: int, inputs: Path) -> Path:
  """Compile the adapter, cpp/main.cpp and any generated C, and link them with the backend's libraries into `dir/bin/benchmark`."""
  out = kernel.dir / "bin"
  shutil.rmtree(out, ignore_errors=True)
  out.mkdir()
  (out / "layout.h").write_text(
    "\n".join(
      [
        "// Input sizes and where each output value sits in the Hessian, for this backend at this horizon.",
        f"#define NZ {problem.n_z(N)}",
        f"#define NP {problem.N_P}",
        f"#define NLAM {problem.n_g(N)}",
        f"#define NNZ {kernel.rows.size}",
        f"static const int ROWS[NNZ] = {{{', '.join(map(str, kernel.rows))}}};",
        f"static const int COLS[NNZ] = {{{', '.join(map(str, kernel.cols))}}};",
        "",
      ]
    )
  )
  flags = [*FLAGS, "-I", CPP, "-I", out, "-I", kernel.dir]
  objects = []
  for source in (*kernel.sources, CPP / kernel.adapter, CPP / "main.cpp"):
    obj = out / f"{source.stem}.o"
    if source.suffix == ".c":
      run(["clang", *flags, "-c", source, "-o", obj])
    else:
      run(["clang++", *flags, "-std=c++17", "-I", gbench() / "include", f'-DINPUTS_DIR="{inputs}"', "-c", source, "-o", obj])
    objects.append(obj)
  binary = out / "benchmark"
  run(["clang++", "-pthread", *objects, "-o", binary, *kernel.libs, gbench() / "lib" / "libbenchmark.a", *VECTOR_LIBM, "-lm"])
  return binary


def time(binary: Path, min_time: str) -> float:
  """CPU time of one evaluation in microseconds, as Google Benchmark reports it; the binary checks its output first."""
  out = subprocess.run([binary, f"--benchmark_min_time={min_time}", "--benchmark_format=json"], check=False, capture_output=True, text=True)
  if out.returncode:
    raise RuntimeError(f"{binary} failed ({out.returncode}): {out.stderr.strip()}")
  data = json.loads(out.stdout[out.stdout.index("{") :])
  return next(b["cpu_time"] for b in data["benchmarks"] if b["run_type"] == "iteration") / 1e3
