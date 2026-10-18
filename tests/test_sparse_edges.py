"""Regression cases at scalar replay, index routing and fallback boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aotax.sparse import sparse_hessian, sparse_jacobian

jax.config.update("jax_enable_x64", True)


def check(f, args, *, hessian, method="structured"):
  result = (sparse_hessian if hessian else sparse_jacobian)(f, *args, method=method)
  reference = np.asarray((jax.hessian if hessian else jax.jacfwd)(f)(*args)).reshape(result.shape)
  got = np.zeros(result.shape)
  values = jax.jit(result.values)(*args)
  assert values.dtype == args[0].dtype
  got[result.rows, result.cols] = values
  tol = 2e-6 if args[0].dtype == np.float32 else 1e-11
  np.testing.assert_allclose(got, np.tril(reference) if hessian else reference, rtol=tol, atol=tol)
  return result


@pytest.mark.parametrize("hessian", [False, True])
@pytest.mark.parametrize("kind", ["overflow", "float32", "integer_parameter"])
def test_scalar_replay_preserves_dtypes(hessian, kind):
  x = np.array([0.2, 0.4, 0.7], dtype=np.float32 if kind == "float32" else np.float64)
  if kind == "overflow":

    def f(x):
      scale = jnp.int32(2147483647) + jnp.int32(1)
      return (x * scale).sum() if not hessian else (x * x * scale).sum()

    args = (x,)
  elif kind == "float32":
    f = lambda x: jnp.sin(2 * x).sum()
    args = (x,)
  else:
    f = lambda x, p: (jnp.sin(x) * jnp.where((p * jnp.int32(3)) % jnp.int32(2) == 0, 2.0, -1.0)).sum()
    args = x, np.array([2147483647, 2, 3], np.int32)
  check(f, args, hessian=hessian)


@pytest.mark.parametrize("weak", [False, True])
def test_scalar_replay_preserves_constant_weak_type(weak):
  from aotax._sparse_scalar import scalarize

  def f(x):
    return jax.lax.convert_element_type_p.bind(np.float32(2), new_dtype=np.dtype("float32"), weak_type=weak, sharding=None)

  spec = jax.ShapeDtypeStruct((1,), np.float32)
  original = jax.eval_shape(f, spec)
  replayed = jax.eval_shape(scalarize(f, spec), (jax.ShapeDtypeStruct((), np.float32),))
  assert replayed.dtype == original.dtype
  assert replayed.weak_type == original.weak_type


@pytest.mark.parametrize("hessian", [False, True])
@pytest.mark.parametrize("mode", ["fill", "clip", "promise_in_bounds"])
@pytest.mark.parametrize("fill", [0.0, 2.0, -3.0])
def test_gather_out_of_bounds_routing(hessian, mode, fill):
  f = lambda x: jnp.sin(x.at[np.array([0, 5, -8])].get(mode=mode, fill_value=fill)).sum()
  if mode == "promise_in_bounds":
    with pytest.raises(ValueError, match="gather.*out.of.bounds.*promise"):
      (sparse_hessian if hessian else sparse_jacobian)(f, np.array([0.4, 0.7, -0.2]), method="structured")
  else:
    check(f, (np.array([0.4, 0.7, -0.2]),), hessian=hessian)


@pytest.mark.parametrize("mode", ["clip", "fill", "promise_in_bounds"])
def test_asdex_rejects_out_of_bounds_gather(mode):
  pytest.importorskip("asdex")
  f = lambda x: jnp.sin(x.at[np.array([0, 5])].get(mode=mode, fill_value=0.0)).sum()
  with pytest.raises(ValueError, match="asdex.*gather.*out.of.bounds"):
    sparse_hessian(f, np.array([0.4, 0.7, -0.2]), method="asdex")


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_, np.complex128])
@pytest.mark.parametrize("method", ["asdex", "auto"])
@pytest.mark.parametrize("hessian", [False, True])
def test_asdex_mixed_parameter_dtypes(dtype, method, hessian):
  pytest.importorskip("asdex")

  # dynamic_slice forces auto through the asdex fallback as well.
  def f(x, p, i):
    y = jax.lax.dynamic_slice(x, (i,), (2,)) ** 2 * jnp.real(p)
    return y.sum() if hessian else jnp.tile(y, 2)

  args = np.array([0.2, 0.4, 0.7]), np.ones(2, dtype), np.array(1, np.int32)
  check(f, args, hessian=hessian, method=method)


def test_deep_graph_budget_precedes_recursion():
  pytest.importorskip("asdex")

  def f(x):
    for _ in range(260):
      x = jnp.sin(x + 0.01)
    return x

  args = (np.array([0.1, 0.2, 0.3]),)
  with pytest.raises(ValueError, match="structured derivatives:.*sin.*budget"):
    sparse_jacobian(f, *args, method="structured")
  result = check(f, args, hessian=False, method="auto")
  assert "fallback:" in result.report and "budget" in result.report


def test_cond_hessian_reports_unsupported_primitive():
  pytest.importorskip("asdex")
  f = lambda x: jax.lax.cond(x[0] > 0, lambda z: jnp.sin(z).sum(), lambda z: z.sum() ** 3, x)
  for method in ("auto", "asdex"):
    with pytest.raises(ValueError, match="asdex.*cond.*unsupported"):
      sparse_hessian(f, np.array([0.2, 0.3]), method=method)


def test_scalar_while_jacobian_uses_forward_mode():
  pytest.importorskip("asdex")

  def f(x):
    return jax.lax.while_loop(lambda s: s[0] < 2, lambda s: (s[0] + 1, jnp.sin(s[1]) + x), (0, x))[1].sum()

  for method in ("auto", "asdex"):
    result = check(f, (np.array([0.2, 0.3]),), hessian=False, method=method)
    assert "asdex fwd" in result.report


def test_shared_coordinate_assembly_storage_is_linear():
  n = 256
  f = lambda x: jnp.sin(x[0] + x[1:]).sum()
  args = (np.linspace(0.1, 0.5, n + 1),)
  result = check(f, args, hessian=True)
  closed = jax.make_jaxpr(result.values)(*args)
  assert sum(np.asarray(c).nbytes for c in closed.consts) < 128 * n


@pytest.mark.parametrize("mode", ["clip", "drop", "promise_in_bounds"])
def test_asdex_rejects_out_of_bounds_scatter(mode):
  pytest.importorskip("asdex")
  f = lambda x: jnp.sin(jnp.zeros(3).at[np.array([0, 5])].add(x, mode=mode)).sum()
  for method in ("auto", "asdex"):
    with pytest.raises(ValueError, match="asdex.*scatter.*out.of.bounds"):
      sparse_hessian(f, np.array([0.4, 0.7]), method=method)
