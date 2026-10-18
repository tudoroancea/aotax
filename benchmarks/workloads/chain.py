"""Chain Hessian and constraint Jacobian, swept over masses at horizon 40.

Uses Scaly's synthetic sweep samples rather than optional harvested closed-loop inputs.
"""

from functools import partial

import asdex
import numpy as np
from workload import Inputs, SparseJax, Workload
from workloads import _chain as problem


def inputs(M: int) -> Inputs:
  z, p = problem.sample_inputs(M, problem.HORIZON)
  return {"z": z, "p": p, "lam_f": np.array(1.0), "lam_g": np.linspace(-0.75, 0.75, problem.n_state(M) * (problem.HORIZON + 1))}


def jac_inputs(M: int) -> Inputs:
  z, p = problem.sample_inputs(M, problem.HORIZON)
  return {"z": z, "p": p}


def reference(M: int, values: Inputs) -> np.ndarray:
  return problem.dense_hessian(M, **values)


def jac_reference(M: int, values: Inputs) -> np.ndarray:
  return problem.dense_jacobian(M, **values)


def scaly_lower(M: int):
  fn = problem.nlp(M).hess("lower")
  sparsity = fn.output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def scaly_jac(M: int):
  fn = problem.nlp(M).jac
  assert fn is not None
  sparsity = fn.output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def aotax_lower(M: int) -> SparseJax:
  fn = partial(problem.lagrangian_jax, M=M)
  shapes = [np.zeros_like(value) for value in inputs(M).values()]
  coloring = asdex.hessian_coloring(fn, *shapes, argnums=0)
  compressed = asdex.compressed_hessian_from_coloring(fn, coloring)
  rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
  lower = np.flatnonzero(rows >= cols)

  def values(z, p, lam_f, lam_g):
    return asdex.decompress_data(compressed(z, p, lam_f, lam_g), coloring)[lower]

  return values, rows[lower], cols[lower]


def aotax_jac(M: int) -> SparseJax:
  fn = partial(problem.constraints_jax, M=M)
  shapes = [np.zeros_like(value) for value in jac_inputs(M).values()]
  coloring = asdex.jacobian_coloring(fn, *shapes, argnums=0)
  compressed = asdex.compressed_jacobian_from_coloring(fn, coloring)

  def values(z, p):
    return asdex.decompress_data(compressed(z, p), coloring)

  return values, np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)


WORKLOADS = (
  Workload("chain", (3, 9, 17), inputs, reference, scaly_lower, aotax_lower),
  Workload("chain_jac", (3, 9, 17), jac_inputs, jac_reference, scaly_jac, aotax_jac, lower_triangle=False),
)
