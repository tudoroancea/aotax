"""Untrained decoder width sweeps at horizon 12, matching Scaly's seed-0 cells."""

import jax
import numpy as np
from workload import Inputs, Workload
from workloads._npmpc_decoder import Decoder


def inputs(width: int) -> Inputs:
  return Decoder(width).sample()


def jac_inputs(width: int) -> Inputs:
  values = inputs(width)
  return {name: values[name] for name in ("z", "p")}


def reference(width: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.hessian(Decoder(width).lagrangian))(*values.values()))


def jac_reference(width: int, values: Inputs) -> np.ndarray:
  return np.asarray(jax.jit(jax.jacfwd(Decoder(width).constraints))(*values.values()))


def scaly_lower(width: int):
  fn = Decoder(width).nlp().hess("lower")
  sparsity = fn.output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def scaly_jac(width: int):
  fn = Decoder(width).nlp().jac
  sparsity = fn.output_sparsities[0]
  assert sparsity is not None
  return fn, np.asarray(sparsity.rows), np.asarray(sparsity.cols)


def function(width: int):
  return Decoder(width).lagrangian


def jac_function(width: int):
  return Decoder(width).constraints


WORKLOADS = (
  Workload("npmpc_decoder", (16, 64, 256), inputs, reference, scaly_lower, function),
  Workload("npmpc_decoder_jac", (16, 64, 256), jac_inputs, jac_reference, scaly_jac, jac_function, lower_triangle=False),
)
