"""IREE's C runtime, built once from source at the compiler's release with only the single-threaded local-sync driver."""

import os
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from importlib.metadata import version
from pathlib import Path

import cmake
import ninja

# In link order: the loader for dispatches linked into the program, the runtime as one archive, and
# the flatbuffer reader the runtime uses for module metadata.
LIBRARIES = ("iree_hal_local_loaders_static_library_loader", "iree_runtime_unified", "flatcc_parsing")


def cache_dir() -> Path:
  return Path(os.environ.get("AOTAX_CACHE", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "aotax"))


def _run(cmd: list, cwd: Path | None = None) -> None:
  subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


@cache
def runtime() -> Path:
  """The install prefix holding the runtime's headers and the static libraries in `LIBRARIES`.

  The PyPI runtime wheel has the same code sealed inside its Python extension, so it is built from
  source, at the tag of the installed compiler: modules only load in a runtime of the same release.
  The compiler, and with it LLVM, is left out, so this takes about a minute.
  """
  tag = f"v{version('iree-base-compiler')}"
  prefix = cache_dir() / f"iree-runtime-{tag}-{sys.platform}"
  if prefix.exists():
    return prefix
  prefix.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=prefix.parent) as tmp:
    src, build, staged = Path(tmp) / "src", Path(tmp) / "build", Path(tmp) / "prefix"
    _run(["git", "clone", "--quiet", "--depth=1", "--branch", tag, "https://github.com/iree-org/iree.git", src])
    submodules = (src / "build_tools" / "scripts" / "git" / "runtime_submodules.txt").read_text().split()
    _run(["git", "submodule", "update", "--init", "--depth=1", "--quiet", *submodules], cwd=src)
    cmake_bin = Path(cmake.CMAKE_BIN_DIR) / "cmake"
    _run(
      [
        cmake_bin,
        "-S",
        src,
        "-B",
        build,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={Path(ninja.BIN_DIR) / 'ninja'}",
        "-DCMAKE_BUILD_TYPE=Release",
        # So that the archives also link into the shared library `Library.load` builds.
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DCMAKE_INSTALL_PREFIX={staged}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DIREE_BUILD_COMPILER=OFF",
        "-DIREE_BUILD_TESTS=OFF",
        "-DIREE_BUILD_SAMPLES=OFF",
        # Creating a runtime instance registers every driver it was built with.
        "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
        "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
      ]
    )
    _run([cmake_bin, "--build", build])
    for component in ("IREEDevLibraries-Runtime", "IREEBundledLibraries"):
      _run([cmake_bin, "--install", build, "--component", component])
    # Renamed into place only once complete, so an interrupted build is never mistaken for one.
    shutil.move(staged, prefix)
  return prefix
