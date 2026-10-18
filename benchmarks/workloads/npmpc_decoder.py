"""Untrained decoder width sweeps at horizon 12, matching Scaly's seed-0 cells."""

import asdex
import jax
import numpy as np
from workload import Inputs, SparseJax, Workload
from workloads import _npmpc as base
from workloads._npmpc_decoder import Decoder, N


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


def aotax_lower(width: int) -> SparseJax:
  fn = Decoder(width).lagrangian
  shapes = tuple(np.zeros_like(value) for value in inputs(width).values())
  coloring = asdex.hessian_coloring(fn, *shapes, argnums=0)
  compressed = asdex.compressed_hessian_from_coloring(fn, coloring)
  rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
  lower = np.flatnonzero(rows >= cols)

  def values(z, p, lam_f, lam_g):
    return asdex.decompress_data(compressed(z, p, lam_f, lam_g), coloring)[lower]

  return values, rows[lower], cols[lower]


def aotax_jac(width: int) -> SparseJax:
  fn = Decoder(width).constraints
  coloring = asdex.jacobian_coloring(fn, np.zeros(base.n_z(N)), np.zeros(Decoder(width).n_p), argnums=0)
  compressed = asdex.compressed_jacobian_from_coloring(fn, coloring)

  def values(z, p):
    return asdex.decompress_data(compressed(z, p), coloring)

  return values, np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)


WORKLOADS = (
  Workload("npmpc_decoder", (16, 64, 256), inputs, reference, scaly_lower, aotax_lower),
  Workload("npmpc_decoder_jac", (16, 64, 256), jac_inputs, jac_reference, scaly_jac, aotax_jac, lower_triangle=False),
)
