"""The lower triangle of the npmpc Lagrangian Hessian with respect to z, from asdex on the JAX Lagrangian and from Scaly."""

from collections.abc import Callable
from functools import partial

import asdex
import jax
import numpy as np
import problem
import scaly as sc


def asdex_lower(N: int) -> tuple[Callable[[jax.Array, jax.Array, jax.Array], jax.Array], np.ndarray, np.ndarray]:
  """A JAX function of (z, p, lam_g) returning the lower-triangle values, and their rows and columns.

  asdex detects the sparsity from the vmapped program, star-colors it (6 colors at every horizon),
  evaluates one Hessian-vector product per color, then gathers the values. It has no option for one
  triangle, so the lower one is selected from its output; the products are the same either way.
  """
  lagrangian = partial(problem.lagrangian_jax, N=N)
  shapes = np.zeros(problem.n_z(N)), np.zeros(problem.N_P), np.zeros(problem.n_g(N))
  coloring = asdex.hessian_coloring(lagrangian, *shapes, argnums=0)
  compressed = asdex.compressed_hessian_from_coloring(lagrangian, coloring)
  rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
  lower = np.flatnonzero(rows >= cols)

  def npmpc_hess(z: jax.Array, p: jax.Array, lam_g: jax.Array) -> jax.Array:
    return asdex.decompress_data(compressed(z, p, lam_g), coloring)[lower]

  return npmpc_hess, rows[lower], cols[lower]


def scaly_lower(N: int) -> tuple[sc.Function, np.ndarray, np.ndarray]:
  """Scaly's sparse Hessian of the Lagrangian Function, lower triangle, and its rows and columns."""
  hessian = sc.sparse_hessian(problem.lagrangian_scaly(N), of="lag", wrt="z", triangle="lower", name="npmpc_hess")
  sparsity = hessian.instantiate().output_sparsities[0]
  assert sparsity is not None
  return hessian, np.asarray(sparsity.rows), np.asarray(sparsity.cols)
