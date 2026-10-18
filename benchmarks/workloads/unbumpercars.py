"""Discrete learned-car safety filter Hessian and constraint Jacobian, swept over car count."""

from functools import partial

import asdex
import jax
import numpy as np
import scaly as sc
from workload import Inputs, SparseJax, SparseScaly, Workload
from workloads import _unbumpercars as problem


def inputs(C: int) -> Inputs:
  return problem.sample_inputs(C)


def jac_inputs(C: int) -> Inputs:
  return {name: value for name, value in inputs(C).items() if name not in ("lam_f", "lam_g")}


def reference(C: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.hessian(partial(problem.lagrangian_jax, C=C)))(*values.values()))


def jac_reference(C: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.jacfwd(partial(problem.constraints_jax, C=C)))(*values.values()))


def scaly_kernel(C: int, kind: str) -> SparseScaly:
  """Use the same oracles as Scaly's solver descriptor, without loading its solver plugin."""
  form = problem.scaly_problem(C)._nlp
  kernel = form.hess("lower") if kind == "hess" else form.jac
  assert kernel is not None
  sparsity = kernel.output_sparsities[0]
  assert sparsity is not None
  return sc.Function._from_instance(kernel), np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def scaly_lower(C: int) -> SparseScaly:
  return scaly_kernel(C, "hess")


def scaly_jac(C: int) -> SparseScaly:
  return scaly_kernel(C, "jac")


def asdex_lower(C: int) -> SparseJax:
  fn = partial(problem.lagrangian_jax, C=C)
  shapes = tuple(np.zeros_like(v) for v in inputs(C).values())
  coloring = asdex.hessian_coloring(fn, *shapes, argnums=0)
  compressed = asdex.compressed_hessian_from_coloring(fn, coloring)
  rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
  lower = np.flatnonzero(rows >= cols)

  def values(*args):
    return asdex.decompress_data(compressed(*args), coloring)[lower]

  return values, rows[lower], cols[lower]


def aotax_jac(C: int) -> SparseJax:
  fn = partial(problem.constraints_jax, C=C)
  shapes = tuple(np.zeros_like(v) for v in jac_inputs(C).values())
  coloring = asdex.jacobian_coloring(fn, *shapes, argnums=0)
  compressed = asdex.compressed_jacobian_from_coloring(fn, coloring)

  def values(*args):
    return asdex.decompress_data(compressed(*args), coloring)

  return values, np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)


WORKLOADS = (
  Workload("unbumpercars", (2, 4, 8), inputs, reference, scaly_lower, asdex_lower),
  Workload("unbumpercars_jac", (2, 4, 8), jac_inputs, jac_reference, scaly_jac, aotax_jac, lower_triangle=False),
)
