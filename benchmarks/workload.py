"""Workload definitions and discovery."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import jax
import numpy as np
import scaly as sc

jax.config.update("jax_enable_x64", True)

Inputs = dict[str, np.ndarray]
SparseScaly = tuple[sc.Function, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Workload:
  """Inputs retain insertion order, which must match both kernels' argument order."""

  name: str
  sizes: tuple[int, ...]
  inputs: Callable[[int], Inputs]
  reference: Callable[[int, Inputs], np.ndarray]
  scaly: Callable[[int], SparseScaly]
  function: Callable[[int], Callable[..., Any]]
  lower_triangle: bool = True


@dataclass(frozen=True)
class ProgramWorkload:
  """Complete JAX programs, with positional pytree arguments and per-leaf (rtol, atol)."""

  name: str
  sizes: tuple[str, ...]
  axes: Callable[[str], dict[str, int]]
  inputs: Callable[[str], tuple[Any, ...]]
  function: Callable[[str], Callable[..., Any]]
  tolerances: Callable[[str], tuple[tuple[float, float], ...]]


def discover() -> dict[str, Workload | ProgramWorkload]:
  workloads = {}
  for path in sorted((Path(__file__).parent / "workloads").glob("*.py")):
    if path.name.startswith("_"):
      continue
    for workload in import_module(f"workloads.{path.stem}").WORKLOADS:
      if workload.name in workloads:
        raise ValueError(f"duplicate workload {workload.name}")
      workloads[workload.name] = workload
  return workloads
