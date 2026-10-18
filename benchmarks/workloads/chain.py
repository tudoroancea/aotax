"""Chain Hessian and constraint Jacobian, swept over masses at horizon 40.

Uses Scaly's synthetic sweep samples rather than optional harvested closed-loop inputs.
"""

from functools import partial

import numpy as np
from workload import Inputs, Workload
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


def function(M: int):
  return partial(problem.lagrangian_jax, M=M)


def jac_function(M: int):
  return partial(problem.constraints_jax, M=M)


WORKLOADS = (
  Workload("chain", (3, 9, 17), inputs, reference, scaly_lower, function),
  Workload("chain_jac", (3, 9, 17), jac_inputs, jac_reference, scaly_jac, jac_function, lower_triangle=False),
)
