import subprocess
import sys
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aotax.sparse import sparse_hessian, sparse_jacobian

jax.config.update("jax_enable_x64", True)


def check_sparse(result, f, args, *, hessian, argnums=0):
  reference = (jax.hessian if hessian else jax.jacfwd)(f, argnums=argnums)(*args)
  reference = np.asarray(reference).reshape(result.shape)
  got = np.zeros(result.shape)
  got[result.rows, result.cols] = jax.jit(result.values)(*args)
  np.testing.assert_allclose(got, np.tril(reference) if hessian else reference, atol=1e-11, rtol=1e-11)
  keys = result.rows * result.shape[1] + result.cols
  assert np.all(keys[1:] > keys[:-1])


@pytest.mark.parametrize("hessian", [False, True])
def test_asdex_windows_and_runtime_weights(hessian):
  pytest.importorskip("asdex")

  def f(p, x, lam):
    y = p * jnp.sin(x[:-1] + x[1:])
    return lam @ y if hessian else y.reshape(2, 3)

  x = np.linspace(-1, 1, 7)
  args = (np.zeros(6), x, np.zeros(6))
  builder = sparse_hessian if hessian else sparse_jacobian
  result = builder(f, *(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args), argnums=1, method="asdex")
  for p, lam in [(np.zeros(6), np.zeros(6)), (np.arange(6.0), np.linspace(-2, 3, 6))]:
    check_sparse(result, f, (p, x, lam), hessian=hessian, argnums=1)
  assert "asdex" in result.report
  assert not result.rows.flags.writeable
  with pytest.raises(ValueError):
    result.rows.setflags(write=True)
  with pytest.raises(FrozenInstanceError):
    result.report = "changed"  # ty: ignore[invalid-assignment]


def test_asdex_duplicate_gather_and_nonlinear_sum():
  pytest.importorskip("asdex")
  x = np.arange(5.0) / 10
  for f in [lambda x: jnp.sin(x[np.array([[0, 0], [1, 3], [3, 4]])].sum(axis=1)).sum(), lambda x: jnp.sin(x).sum() ** 2]:
    result = sparse_hessian(f, x, method="asdex")
    check_sparse(result, f, (x,), hessian=True)


def test_validation():
  with pytest.raises(ValueError, match="method"):
    sparse_hessian(jnp.sum, np.ones(3), method="dense")
  with pytest.raises(ValueError, match="argnums"):
    sparse_hessian(jnp.sum, np.ones(3), argnums=(0,))  # ty: ignore[invalid-argument-type]
  with pytest.raises(TypeError, match="real floating"):
    sparse_hessian(jnp.sum, np.ones(3, dtype=int))
  with pytest.raises(ValueError, match="scalar"):
    sparse_hessian(lambda x: x, np.ones(3))
  with pytest.raises(TypeError, match="one real floating"):
    sparse_jacobian(lambda x: (x, x), np.ones(3))


def test_lazy_import():
  subprocess.run([sys.executable, "-c", "import aotax, aotax.sparse, sys; assert 'asdex' not in sys.modules"], check=True)
