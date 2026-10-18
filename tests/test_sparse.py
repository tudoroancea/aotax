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


@pytest.mark.parametrize("hessian", [False, True])
@pytest.mark.parametrize("layout", ["overlap", "strided", "gather", "duplicates"])
def test_structured_windows(hessian, layout):
  x = np.linspace(-0.7, 0.9, 12)
  indices = {
    "overlap": np.arange(4)[:, None] + np.arange(3),
    "strided": np.arange(4)[:, None] * 2 + np.arange(3) * 2,
    "gather": np.array([[9, 2, 4], [4, 7, 1], [2, 0, 11]]),
    "duplicates": np.array([[0, 0, 3], [2, 2, 2], [7, 3, 7]]),
  }[layout]

  def f(x, p, lam):
    window = (
      x[indices]
      if layout in {"gather", "duplicates"}
      else (jnp.stack([x[:8:2], x[2:10:2], x[4:12:2]], axis=1) if layout == "strided" else jnp.stack([x[:4], x[1:5], x[2:6]], axis=1))
    )
    y = jnp.sin(window.sum(axis=1) * p)
    return lam @ y if hessian else y

  count = len(indices)
  args = x, np.zeros(count), np.zeros(count)
  builder = sparse_hessian if hessian else sparse_jacobian
  result = builder(f, *args, method="structured")
  for seed in range(3):
    rng = np.random.default_rng(seed)
    changed = x + rng.normal(scale=0.1, size=x.size), rng.normal(size=count), rng.normal(size=count)
    check_sparse(result, f, changed, hessian=hessian)
  assert "local maps" in result.report
  assert "input map" in result.report


@pytest.mark.parametrize("hessian", [False, True])
def test_selected_broadcast_global_and_parameter_windows(hessian):
  def f(p, x, weights):
    y = jnp.sin(x[1:] + x[0] * p) * weights
    return y.sum() if hessian else y

  args = np.linspace(0.1, 0.6, 8), np.linspace(-1, 1, 9), np.zeros(8)
  result = (sparse_hessian if hessian else sparse_jacobian)(f, *args, argnums=-2, method="structured")
  check_sparse(result, f, (*args[:2], np.arange(8.0)), hessian=hessian, argnums=1)
  assert np.any(result.cols == 0)


def test_nonlinear_outer_reduction_and_mixed_axes():
  x = np.linspace(-0.6, 0.8, 12).reshape(3, 4)
  for f in [lambda x: jnp.sin(x).sum() ** 2, lambda x: jnp.sum(jnp.sin(x.sum(axis=0)) ** 2), lambda x: jnp.sum((x @ x.T) ** 2)]:
    result = sparse_hessian(f, x, method="structured")
    check_sparse(result, f, (x,), hessian=True)
  assert result.rows.size == x.size * (x.size + 1) // 2


def test_matrix_product_stage(tmp_path):
  import aotax

  def f(x, w, lam):
    h = jnp.exp(x.reshape(4, 3) @ w)
    return (lam * jnp.sin(h @ w.T)).sum()

  args = np.linspace(-0.3, 0.4, 12), np.arange(15.0).reshape(3, 5) / 20, np.ones((4, 3))
  result = sparse_hessian(f, *args, method="structured")
  check_sparse(result, f, args, hessian=True)
  native = aotax.compile(result.values, *args, out=tmp_path).load()
  np.testing.assert_allclose(native(*args), result.values(*args), rtol=1e-11, atol=1e-11)
  assert "stablehlo.dot_general" in (tmp_path / "iree/stablehlo.mlir").read_text()


def test_custom_jvp_is_preserved():
  @jax.custom_jvp
  def f(x):
    return jnp.sin(x)

  @f.defjvp
  def rule(primals, tangents):
    (x,), (dx,) = primals, tangents
    return f(x), 3 * x * dx

  x = np.linspace(-1, 1, 9)
  for hessian, function in [(False, f), (True, lambda x: f(x).sum())]:
    result = (sparse_hessian if hessian else sparse_jacobian)(function, x, method="structured")
    check_sparse(result, function, (x,), hessian=hessian)


def test_custom_jvp_fallback_and_strict_error():
  pytest.importorskip("asdex")

  @jax.custom_jvp
  def f(x):
    return jnp.sin(x)

  @f.defjvp
  def rule(primals, tangents):
    (x,), (dx,) = primals, tangents
    return f(x), jnp.broadcast_to(dx.sum(), x.shape) * x

  x = np.linspace(-0.5, 0.5, 5)
  with pytest.raises(ValueError, match="custom_jvp_call.*reduce_sum"):
    sparse_jacobian(f, x, method="structured")
  for method in ["auto", "asdex"]:
    result = sparse_jacobian(f, x, method=method)
    check_sparse(result, f, (x,), hessian=False)
  assert "pullback" in result.report


def test_custom_vjp_fallback():
  pytest.importorskip("asdex")

  @jax.custom_vjp
  def f(x):
    return jnp.sin(x)

  f.defvjp(lambda x: (f(x), x), lambda x, cotangent: (4 * x * cotangent,))
  x = np.linspace(-1, 1, 6)
  with pytest.raises(ValueError, match="custom_vjp_call"):
    sparse_jacobian(f, x, method="structured")
  result = sparse_jacobian(f, x)
  reference = np.asarray(jax.jacrev(f)(x))
  got = np.zeros(result.shape)
  got[result.rows, result.cols] = result.values(x)
  np.testing.assert_allclose(got, reference)
  assert "custom_vjp_call" in result.report


def test_affine_and_constant():
  x = np.linspace(-1, 1, 6)
  result = sparse_jacobian(lambda x: jnp.concatenate([x[1:] - x[:-1], -x, x[:1] * 3]), x, method="structured")
  check_sparse(result, lambda x: jnp.concatenate([x[1:] - x[:-1], -x, x[:1] * 3]), (x,), hessian=False)
  assert "affine coefficients" in result.report
  for builder in [sparse_hessian, sparse_jacobian]:
    result = builder(lambda x: jnp.array(2.0), x, method="structured")
    assert result.rows.size == result.values(x).size == 0


def test_control_flow_report():
  pytest.importorskip("asdex")

  def f(x):
    return jax.lax.scan(lambda a, b: (a + jnp.sin(b), a), np.array(0.0), x)[1]

  x = np.linspace(-1, 1, 5)
  with pytest.raises(ValueError, match="scan.*backward"):
    sparse_jacobian(f, x, method="structured")
  result = sparse_jacobian(f, x)
  assert "fallback: scan" in result.report
  check_sparse(result, f, (x,), hessian=False)


def test_nonsymmetric_custom_second_derivatives():
  pytest.importorskip("asdex")

  @jax.custom_jvp
  def f(x):
    return jnp.sum(x * x)

  @f.defjvp
  def rule(primals, tangents):
    (x,), (dx,) = primals, tangents
    return f(x), jnp.sum(x[::-1] * dx * jnp.arange(1, x.size + 1))

  x = np.arange(4.0)
  for method in ["auto", "asdex"]:
    result = sparse_hessian(f, x, method=method)
    check_sparse(result, f, (x,), hessian=True)


def test_empty_output():
  result = sparse_jacobian(lambda x: x[:0], np.zeros(3), method="structured")
  assert result.shape == (0, 3)
  assert result.values(np.zeros(3)).shape == (0,)


@pytest.mark.parametrize("method", ["structured", "asdex"])
@pytest.mark.parametrize("hessian", [False, True])
def test_row_major_array_argument(method, hessian):
  if method == "asdex":
    pytest.importorskip("asdex")

  def f(p, x):
    y = p * jnp.sin(x.T)
    return y.sum() if hessian else y

  args = np.arange(6.0).reshape(3, 2), np.linspace(-1, 1, 6).reshape(2, 3)
  result = (sparse_hessian if hessian else sparse_jacobian)(f, *args, argnums=1, method=method)
  check_sparse(result, f, args, hessian=hessian, argnums=1)


def test_batched_contraction():
  def f(x, p):
    return jnp.sin(jnp.einsum("bij,bjk->bik", x.reshape(2, 3, 4), p)).sum()

  args = np.linspace(-0.8, 0.9, 24), np.linspace(-0.5, 0.8, 16).reshape(2, 4, 2)
  result = sparse_hessian(f, *args, method="structured")
  check_sparse(result, f, args, hessian=True)


@pytest.mark.parametrize("hessian", [False, True])
def test_shared_primal_outputs_are_grouped(hessian):
  def f(x, p, lam):
    x = x.reshape(13, 3)
    h = jnp.sin(x[:, 0] * x[:, 1] + x[:, 2])
    y = jnp.stack([h * h, jnp.exp(h)], axis=1) * p
    return (lam * y).sum() if hessian else y

  args = np.linspace(-0.5, 0.6, 39), np.array([0.2, -0.3]), np.zeros((13, 2))
  result = (sparse_hessian if hessian else sparse_jacobian)(f, *args, method="structured")
  maps = [line for line in result.report.splitlines() if "local maps" in line]
  assert len(maps) == 1
  assert "13 local maps, width 3, 2 outputs/terms" in maps[0]
  assert ("hessian" if hessian else "jacrev") in maps[0]
  check_sparse(result, f, (*args[:2], np.linspace(-1, 1, 26).reshape(13, 2)), hessian=hessian)


def test_forward_mode_and_shared_matrix_parameters():
  def f(x, w):
    h = jnp.sin(x.reshape(17, 2) @ w)
    return jnp.stack([h[:, 0], h[:, 1], h[:, 0] * h[:, 1]], axis=1)

  args = np.linspace(-0.5, 0.5, 34), np.linspace(-0.3, 0.4, 4).reshape(2, 2)
  result = sparse_jacobian(f, *args, method="structured")
  check_sparse(result, f, args, hessian=False)
  assert "17 local maps, width 2, 3 outputs/terms" in result.report
  assert "jacfwd" in result.report
  closed = jax.make_jaxpr(result.values)(*args)
  products = [eq for eq in closed.jaxpr.eqns if eq.primitive.name == "dot_general"]
  assert products
  assert all(not any(eq.params["dimension_numbers"][1]) for eq in products)
  assert all(any(getattr(v.aval, "shape", None) == (2, 2) for v in eq.invars) for eq in products)


def test_singleton_array_dot_and_reduced_jacobian():
  x = np.linspace(-0.6, 0.7, 6).reshape(3, 2)
  p = np.array([0.4, 0.8])
  for f in [lambda x, p: jnp.sin(x[:1] @ p), lambda x, p: jnp.sin(x * p).sum(axis=0)]:
    result = sparse_jacobian(f, x, p, method="structured")
    check_sparse(result, f, (x, p), hessian=False)


def test_wide_scalar_hessian_uses_construction_budget():
  pytest.importorskip("asdex")

  def f(x):
    h = x.reshape(3, 12)
    for _ in range(12):
      h = jnp.sin(h + 0.1 * jnp.roll(h, 1, axis=1))
    return h.sum()

  x = np.linspace(-0.3, 0.4, 36)
  with pytest.raises(ValueError, match="local Hessian width.*nested sparse"):
    sparse_hessian(f, x, method="structured")
  result = sparse_hessian(f, x)
  assert "fallback:" in result.report
  assert "local Hessian width" in result.report
  check_sparse(result, f, (x,), hessian=True)


@pytest.mark.parametrize("hessian", [False, True])
def test_transformed_shared_and_moving_parameter_windows(hessian):
  def f(x, p):
    q = jnp.sin(p)
    h = jnp.sin(x.reshape(11, 2) @ q[-4:].reshape(2, 2) + q[:11, None])
    y = jnp.stack([h[:, 0], h[:, 1], h[:, 0] * h[:, 1]], axis=1)
    return jnp.sum(y * y) if hessian else y

  args = np.linspace(-0.4, 0.5, 22), np.linspace(-0.7, 0.8, 15)
  result = (sparse_hessian if hessian else sparse_jacobian)(f, *args, method="structured")
  check_sparse(result, f, args, hessian=hessian)
  closed = jax.make_jaxpr(result.values)(*args)
  products = [eq for eq in closed.jaxpr.eqns if eq.primitive.name == "dot_general"]
  assert products
  assert all(not any(eq.params["dimension_numbers"][1]) for eq in products)


@pytest.mark.parametrize("custom", [False, True])
def test_hessian_keeps_primal_values_with_zero_derivative_columns(custom, tmp_path):
  @jax.custom_jvp
  def rule_value(a, b):
    return a * a + b

  @rule_value.defjvp
  def rule(primals, tangents):
    a, b = primals
    da, _db = tangents
    return rule_value(a, b), da * a * b

  def f(x, w):
    z = x.reshape(7, 2)
    y = rule_value(z[:, 0], z[:, 1]) if custom else z[:, 0] ** 2 * jax.lax.stop_gradient(z[:, 1])
    return (jnp.stack([y, y], axis=1) @ w).sum()

  args = np.linspace(-0.4, 0.6, 14), np.array([[0.7], [-0.2]])
  result = sparse_hessian(f, *args, method="structured")
  check_sparse(result, f, args, hessian=True)

  if custom:
    import aotax

    native = aotax.compile(result.values, *args, out=tmp_path).load()
    np.testing.assert_allclose(native(*args), result.values(*args), rtol=1e-11, atol=1e-11)
