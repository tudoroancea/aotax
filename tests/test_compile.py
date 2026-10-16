import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aotax

jax.config.update("jax_enable_x64", True)


def check(fn, *args, tmp_path, rtol=1e-12):
  loaded = aotax.compile(fn, *args, out=tmp_path).load()
  got, want = loaded(*args), jax.jit(fn)(*args)
  jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w, rtol=rtol, atol=rtol), got, want)
  return loaded


def test_f64_transcendentals_and_matmul(tmp_path):
  rng = np.random.default_rng(0)

  def mlp(w, x):
    return jnp.tanh(w @ jnp.sin(x)) + jnp.exp(jnp.cos(x[:3]))

  check(mlp, rng.normal(size=(3, 5)), rng.normal(size=5), tmp_path=tmp_path)


def test_pytrees_scalars_and_mixed_dtypes(tmp_path):
  def step(state, k, scale):
    x, v = state["x"], state["v"]
    return {"x": x + v, "v": v * scale}, (k + 1, jnp.sum(x).astype(jnp.float32))

  state = {"x": np.arange(4.0), "v": np.ones(4)}
  check(step, state, np.int64(7), np.float64(0.5), tmp_path=tmp_path)


def test_f32_and_int32(tmp_path):
  def f(x, n):
    return x * 2.0, n * n

  check(f, np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3), np.arange(3, dtype=np.int32), tmp_path=tmp_path, rtol=1e-6)


def test_hessian_repeated_calls(tmp_path):
  def energy(q):
    return jnp.sum(jnp.cos(q[1:] - q[:-1])) + 0.5 * q @ q

  rng = np.random.default_rng(1)
  loaded = check(jax.hessian(energy), rng.normal(size=8), tmp_path=tmp_path)
  for _ in range(3):
    q = rng.normal(size=8)
    np.testing.assert_allclose(loaded(q), jax.hessian(energy)(q), rtol=1e-12, atol=1e-12)


def test_rejects_bad_inputs(tmp_path):
  def f(x):
    return x + 1.0

  loaded = aotax.compile(f, np.zeros(3), out=tmp_path).load()
  with pytest.raises(ValueError, match="shape"):
    loaded(np.zeros(4))
  with pytest.raises(TypeError, match="nest"):
    loaded(np.zeros(3), np.zeros(3))
  with pytest.raises(TypeError, match="dtype"):
    aotax.compile(f, np.zeros(3, dtype=np.complex128), out=tmp_path / "c")
  with pytest.raises(ValueError, match="C identifier"):
    aotax.compile(lambda x: x, np.zeros(3), out=tmp_path / "l")
