"""Race-car Lagrangian Hessian and constraint Jacobian, swept over horizon."""

from functools import partial

import jax
import numpy as np
from workload import Inputs, Workload
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


def function(N: int):
  return partial(problem.lagrangian_jax, N=N)


def jac_function(N: int):
  return partial(problem.constraints_jax, N=N)


WORKLOADS = (
  Workload("race_cars", (5, 40, 500), inputs, reference, scaly_lower, function),
  Workload("race_cars_jac", (5, 40, 500), jac_inputs, jac_reference, scaly_jac, jac_function, lower_triangle=False),
)
