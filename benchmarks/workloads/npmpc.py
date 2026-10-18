"""Neural-process MPC Hessian and constraint Jacobian, swept over horizon."""

from functools import partial

import asdex
import jax
import numpy as np
import scaly as sc
from workload import Inputs, SparseJax, SparseScaly, Workload
from workloads import _npmpc as problem


def inputs(N: int) -> Inputs:
  return dict(zip(("z", "p", "lam_g"), problem.sample_inputs(N), strict=True))


def jac_inputs(N: int) -> Inputs:
  return {name: value for name, value in inputs(N).items() if name != "lam_g"}


def reference(N: int, values: Inputs) -> np.ndarray:
  return problem.dense_hessian(N, **values)


def jac_reference(N: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.jacfwd(partial(problem.constraints_jax, N=N)))(*values.values()))


def aotax_jac(N: int) -> SparseJax:
  fn = partial(problem.constraints_jax, N=N)
  coloring = asdex.jacobian_coloring(fn, np.zeros(problem.n_z(N)), np.zeros(problem.N_P), argnums=0)
  compressed = asdex.compressed_jacobian_from_coloring(fn, coloring)

  def values(z, p):
    return asdex.decompress_data(compressed(z, p), coloring)

  return values, np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)


def scaly_jac(N: int) -> SparseScaly:
  fn = sc.sparse_jacobian(problem.constraints_scaly(N), of="g", wrt="z", name="npmpc_jac")
  sparsity = fn.instantiate().output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def asdex_lower(N: int) -> SparseJax:
  """asdex sparse Hessian of the JAX Lagrangian, restricted to its lower triangle."""
  lagrangian = partial(problem.lagrangian_jax, N=N)
  shapes = np.zeros(problem.n_z(N)), np.zeros(problem.N_P), np.zeros(problem.n_g(N))
  coloring = asdex.hessian_coloring(lagrangian, *shapes, argnums=0)
  compressed = asdex.compressed_hessian_from_coloring(lagrangian, coloring)
  rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
  lower = np.flatnonzero(rows >= cols)

  def npmpc_hess(z: jax.Array, p: jax.Array, lam_g: jax.Array) -> jax.Array:
    return asdex.decompress_data(compressed(z, p, lam_g), coloring)[lower]

  return npmpc_hess, rows[lower], cols[lower]


def scaly_lower(N: int) -> SparseScaly:
  """Scaly's sparse Hessian of the Lagrangian Function, lower triangle, and its rows and columns."""
  hessian = sc.sparse_hessian(problem.lagrangian_scaly(N), of="lag", wrt="z", triangle="lower", name="npmpc_hess")
  sparsity = hessian.instantiate().output_sparsities[0]
  assert sparsity is not None
  return hessian, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


WORKLOADS = (
  Workload("npmpc", (12, 50, 200), inputs, reference, scaly_lower, asdex_lower),
  Workload("npmpc_jac", (12, 50, 200), jac_inputs, jac_reference, scaly_jac, aotax_jac, lower_triangle=False),
)
