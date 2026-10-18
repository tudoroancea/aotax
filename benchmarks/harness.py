"""Lower sparse workloads, build correctness-gated Google Benchmark binaries, and time them."""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cmake
import jax
import ninja
import numpy as np
from scaly.codegen.abi import c_ident
from scaly.codegen.aot import render_c_module
from scaly.codegen.toolchain import find_c_compiler, native_recipe
from workload import Inputs, Workload

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
  """Generated sources, libraries, sparse coordinates and adapter in `dir`."""

  dir: Path
  rows: np.ndarray
  cols: np.ndarray
  adapter: str
  sources: tuple[Path, ...] = ()
  libs: tuple[str | Path, ...] = ()


def lower_scaly(workload: Workload, size: int, out: Path, inputs: Inputs) -> Kernel:
  """Render the workload's Scaly sparse kernel."""
  function, rows, cols = workload.scaly(size)
  compiler = find_c_compiler()
  vector_libm = native_recipe(compiler.command).vector_libm if compiler is not None else "none"
  module = render_c_module(function, header_name="generated.h", source_name="generated.c", vector_libm=vector_libm)
  (out / module.header_name).write_text(module.header)
  (out / module.source_name).write_text(module.source)
  name = c_ident(function.name)
  adapter = (CPP / "scaly.c").read_text().replace("GENERATED", name)
  (out / "adapter.c").write_text(adapter)
  return Kernel(out, rows, cols, adapter="adapter.c", sources=(out / module.source_name,))


def lower_aotax(workload: Workload, size: int, out: Path, inputs: Inputs) -> Kernel:
  """Compile the workload's sparse JAX kernel."""
  function, rows, cols = workload.aotax(size)
  specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in inputs.values()]
  library = aotax.compile(function, *specs, out=out, name="generated")
  args = ", ".join([*(f"g_inputs[{i}]" for i in range(len(inputs))), "out"])
  adapter = (CPP / "aotax.c").read_text().replace("CALL_ARGS", args)
  (out / "adapter.c").write_text(adapter)
  return Kernel(out, rows, cols, adapter="adapter.c", libs=(library.archive, *aotax.SYSTEM_LIBS))


def write_inputs(directory: Path, workload: Workload, size: int, inputs: Inputs) -> np.ndarray:
  """Write named float64 inputs and an independent dense reference as raw doubles."""
  directory.mkdir(parents=True, exist_ok=True)
  for name, values in inputs.items():
    if values.dtype != np.float64:
      raise ValueError(f"{name}: expected float64, got {values.dtype}")
    values.tofile(directory / f"{name}.bin")
  reference = np.asarray(workload.reference(size, inputs), dtype=np.float64)
  if reference.ndim != 2:
    raise ValueError("reference must be a matrix")
  reference.tofile(directory / "reference.bin")
  return reference


def build(kernel: Kernel, workload: Workload, inputs: Inputs, reference: np.ndarray, directory: Path) -> Path:
  """Compile the adapter and shared correctness/timing binary."""
  out = kernel.dir / "bin"
  out.mkdir()
  (out / "layout.h").write_text(
    "\n".join(
      [
        f"#define NINPUTS {len(inputs)}",
        f"#define NROWS {reference.shape[0]}",
        f"#define NCOLS {reference.shape[1]}",
        f"#define LOWER_TRIANGLE {int(workload.lower_triangle)}",
        f"#define NNZ {kernel.rows.size}",
        f"static const int ROWS[NNZ] = {{{', '.join(map(str, kernel.rows))}}};",
        f"static const int COLS[NNZ] = {{{', '.join(map(str, kernel.cols))}}};",
        f"static const int INPUT_SIZES[NINPUTS] = {{{', '.join(str(v.size) for v in inputs.values())}}};",
        "static const char* INPUT_NAMES[NINPUTS] = {" + ", ".join(json.dumps(n + ".bin") for n in inputs) + "};",
        "",
      ]
    )
  )
  flags = [*FLAGS, "-I", CPP, "-I", out, "-I", kernel.dir]
  objects = []
  for source in (*kernel.sources, kernel.dir / kernel.adapter, CPP / "main.cpp"):
    obj = out / f"{source.stem}.o"
    if source.suffix == ".c":
      run(["clang", *flags, "-c", source, "-o", obj])
    else:
      run(["clang++", *flags, "-std=c++17", "-I", gbench() / "include", f'-DINPUTS_DIR="{directory}"', "-c", source, "-o", obj])
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
