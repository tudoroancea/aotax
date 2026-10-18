"""Neural-process MPC Hessian and constraint Jacobian, swept over horizon."""

from functools import partial

import jax
import numpy as np
import scaly as sc
from workload import Inputs, SparseScaly, Workload
from workloads import _npmpc as problem


def inputs(N: int) -> Inputs:
  return dict(zip(("z", "p", "lam_g"), problem.sample_inputs(N), strict=True))


def jac_inputs(N: int) -> Inputs:
  return {name: value for name, value in inputs(N).items() if name != "lam_g"}


def reference(N: int, values: Inputs) -> np.ndarray:
  return problem.dense_hessian(N, **values)


def jac_reference(N: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.jacfwd(partial(problem.constraints_jax, N=N)))(*values.values()))


def scaly_jac(N: int) -> SparseScaly:
  fn = sc.sparse_jacobian(problem.constraints_scaly(N), of="g", wrt="z", name="npmpc_jac")
  sparsity = fn.instantiate().output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def scaly_lower(N: int) -> SparseScaly:
  """Scaly's sparse Hessian of the Lagrangian Function, lower triangle, and its rows and columns."""
  hessian = sc.sparse_hessian(problem.lagrangian_scaly(N), of="lag", wrt="z", triangle="lower", name="npmpc_hess")
  sparsity = hessian.instantiate().output_sparsities[0]
  assert sparsity is not None
  return hessian, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def function(N: int):
  return partial(problem.lagrangian_jax, N=N)


def jac_function(N: int):
  return partial(problem.constraints_jax, N=N)


WORKLOADS = (
  Workload("npmpc", (12, 50, 200), inputs, reference, scaly_lower, function),
  Workload("npmpc_jac", (12, 50, 200), jac_inputs, jac_reference, scaly_jac, jac_function, lower_triangle=False),
)
