"""Race-car Lagrangian Hessian and constraint Jacobian, swept over horizon."""

from functools import partial

import asdex
import jax
import numpy as np
from workload import Inputs, SparseJax, Workload
from workloads import _race_cars as problem


def inputs(N: int) -> Inputs:
  return problem.sample_inputs(N)


def jac_inputs(N: int) -> Inputs:
  values = inputs(N)
  return {name: values[name] for name in ("z", "p")}


def reference(N: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.hessian(partial(problem.lagrangian_jax, N=N)))(*values.values()))


def jac_reference(N: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.jacfwd(partial(problem.constraints_jax, N=N)))(*values.values()))


def scaly_lower(N: int):
  """The exact lower Hessian used by Scaly's IPOPT descriptor, without loading the solver plugin."""
  fn = problem.problem_scaly(N)._nlp.hess("lower")
  sparsity = fn.output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def scaly_jac(N: int):
  """The full constraint Jacobian used by Scaly's solver descriptor."""
  fn = problem.problem_scaly(N)._nlp.jac
  assert fn is not None
  sparsity = fn.output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def asdex_lower(N: int) -> SparseJax:
  lagrangian = partial(problem.lagrangian_jax, N=N)
  coloring = asdex.hessian_coloring(lagrangian, *inputs(N).values(), argnums=0)
  compressed = asdex.compressed_hessian_from_coloring(lagrangian, coloring)
  rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
  lower = np.flatnonzero(rows >= cols)

  def values(z, p, lam_f, lam_g):
    return asdex.decompress_data(compressed(z, p, lam_f, lam_g), coloring)[lower]

  return values, rows[lower], cols[lower]


def aotax_jac(N: int) -> SparseJax:
  constraints = partial(problem.constraints_jax, N=N)
  coloring = asdex.jacobian_coloring(constraints, *jac_inputs(N).values(), argnums=0)
  compressed = asdex.compressed_jacobian_from_coloring(constraints, coloring)

  def values(z, p):
    return asdex.decompress_data(compressed(z, p), coloring)

  return values, np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)


WORKLOADS = (
  Workload("race_cars", (5, 40, 500), inputs, reference, scaly_lower, asdex_lower),
  Workload("race_cars_jac", (5, 40, 500), jac_inputs, jac_reference, scaly_jac, aotax_jac, lower_triangle=False),
)
