import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aotax
from aotax import _Loaded

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


def test_shared_math_broadcast(tmp_path):
  def f(x, w):
    return jnp.sin(x)[:, None] * w + jnp.cos(x)[None, :] + jnp.tanh(x)[:, None]

  x = np.r_[np.linspace(-20, 20, 65), -1e12, 1e12, -0.0, 0.0]
  w = np.linspace(-1, 1, x.size**2).reshape(x.size, x.size)
  loaded = check(f, x, w, tmp_path=tmp_path)
  assert "stablehlo.optimization_barrier" in (tmp_path / "iree/stablehlo.mlir").read_text()
  np.testing.assert_allclose(loaded(x + 0.125, w), f(x + 0.125, w), rtol=1e-12, atol=1e-12)


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


def test_scan(tmp_path):
  def scan(x, xs):
    return jax.lax.scan(lambda carry, v: (carry * 0.5 + v, carry * v), x, xs)

  loaded = check(scan, np.ones(4), np.arange(28.0).reshape(7, 4), tmp_path=tmp_path)
  for length in (0.0, 3.0):
    x, xs = np.full(4, length), np.arange(28.0).reshape(7, 4) * 0.3
    jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w), loaded(x, xs), scan(x, xs))


def test_while_loop(tmp_path):
  def loop(x, limit):
    return jax.lax.while_loop(lambda v: v[0] < limit, lambda v: (v[0] + 1, v[1] * 1.25 + v[0]), (np.int32(0), x))[1]

  loaded = check(loop, np.arange(5.0), np.int32(9), tmp_path=tmp_path)
  for limit in (0, 1, 3, 12):
    np.testing.assert_allclose(loaded(np.arange(5.0), np.int32(limit)), loop(np.arange(5.0), np.int32(limit)))


def test_cond(tmp_path):
  def choose(x, flag):
    return jax.lax.cond(flag > 0, lambda a: jnp.sin(a), lambda a: a * a, x)

  loaded = check(choose, np.arange(5.0), np.int32(1), tmp_path=tmp_path)
  np.testing.assert_allclose(loaded(np.arange(5.0), np.int32(-1)), np.arange(5.0) ** 2)


def test_constants_and_independent_outputs(tmp_path):
  weights = np.arange(128.0).reshape(16, 8) / 100

  def constant(x):
    return weights @ x, x, x + 2, jnp.asarray(weights)

  check(constant, np.linspace(-1, 1, 8), tmp_path=tmp_path)


@pytest.mark.parametrize("offset", [0, 1])
def test_misalignment_and_aliases(tmp_path, offset):
  def outputs(x):
    return x * 3, x + 7

  loaded = _Loaded(aotax.compile(outputs, np.zeros(8), out=tmp_path))
  raw = np.empty(8 * 8 + 64, np.uint8)
  x = np.ndarray((8,), dtype=np.float64, buffer=raw, offset=(-raw.ctypes.data % 64) + offset)
  x[:] = np.arange(8.0)
  storage = np.empty(8 * 8 + 64, np.uint8)
  out = np.ndarray((8,), dtype=np.float64, buffer=storage, offset=(-storage.ctypes.data % 64) + offset)
  assert loaded._call(loaded._context, x.ctypes.data, out.ctypes.data, x.ctypes.data) == 0
  np.testing.assert_allclose(out, np.arange(8.0) * 3)
  np.testing.assert_allclose(x, np.arange(8.0) + 7)
  x[:] = np.arange(8.0)
  assert loaded._call(loaded._context, x.ctypes.data, out.ctypes.data, out.ctypes.data) == 0
  np.testing.assert_allclose(out, np.arange(8.0) + 7)
  assert loaded._call(None, x.ctypes.data, out.ctypes.data, out.ctypes.data) == 1


def test_multiple_contexts(tmp_path):
  from concurrent.futures import ThreadPoolExecutor

  def compute(x):
    return jnp.sin(x) * x

  library = aotax.compile(compute, np.zeros(32), out=tmp_path)
  contexts = [library.load(), library.load()]
  xs = [np.arange(32.0) + i for i in range(8)]

  def run(context):
    return [context(x) for x in xs]

  with ThreadPoolExecutor(2) as pool:
    results = list(pool.map(run, contexts))
  for got in results:
    for x, result in zip(xs, got, strict=True):
      np.testing.assert_allclose(result, compute(x))


def test_reload_after_recompile(tmp_path):
  x = np.arange(4.0)
  first = aotax.compile(lambda x: x + 1, x, out=tmp_path, name="f").load()
  second = aotax.compile(lambda x: x + 2, x, out=tmp_path, name="f").load()
  np.testing.assert_array_equal(first(x), x + 1)
  np.testing.assert_array_equal(second(x), x + 2)


def test_no_dispatches(tmp_path):
  def identity(x):
    return x, x, x

  check(identity, np.arange(4.0), tmp_path=tmp_path)

  def nothing():
    return ()

  assert aotax.compile(nothing, out=tmp_path).load()() == ()


def test_subnormal_inputs(tmp_path):
  def amplify(x):
    return x * 1e300

  x = np.array([1e-310, -1e-310, 1e-300])
  before = np.multiply(x, 1e300)
  check(amplify, x, tmp_path=tmp_path)
  np.testing.assert_array_equal(np.multiply(x, 1e300), before)


def test_dynamic_indices(tmp_path):
  def index(x, i):
    return jax.lax.dynamic_slice(x, (i,), (3,)), x[i]

  x = np.arange(10.0)
  loaded = check(index, x, np.int32(2), tmp_path=tmp_path)
  for i in (-50, -4, -1, 0, 2, 9, 50):
    got, want = loaded(x, np.int32(i)), jax.jit(index)(x, np.int32(i))
    jax.tree.map(np.testing.assert_array_equal, got, want)


@pytest.mark.parametrize(
  "fn,args",
  [
    (lambda x: x.at[2:10].set(0.0), ()),
    (lambda x: jnp.sum(x.at[2:10].set(0.0)), ()),
    (lambda x, i: x.at[i].set(42.0), (np.int32(3),)),
    (lambda x: x.at[3].set(42.0), ()),
    (lambda x, y, i: jax.lax.dynamic_update_slice(x, y, (i,)), (np.arange(8.0) + 100, np.int32(3))),
    (lambda x, i: x.at[i].multiply(2.0), (np.int32(3),)),
    (lambda x, i: jnp.sin(x.at[i].set(1.0)), (np.int32(3),)),
    (lambda x, y: x.at[2:10].set(y), (np.arange(8.0) + 100,)),
    (lambda x: x.at[2:10].set(x[:8]), ()),
    (lambda x: jax.lax.fori_loop(0, 5, lambda i, v: v.at[1:].set(v[:-1]), x), ()),
    (lambda x: jax.lax.scan(lambda c, _: (c.at[1:].set(c[:-1]), None), x, None, length=3)[0], ()),
  ],
  ids=["fill", "fill_sum", "scatter", "static", "dynamic_update", "multiply", "sin_set", "queue_copy", "self_copy", "loop", "scan"],
)
def test_inputs_unchanged(tmp_path, fn, args):
  raw = np.empty(16 * 8 + 64, np.uint8)
  x = np.ndarray((16,), dtype=np.float64, buffer=raw, offset=-raw.ctypes.data % 64)
  x[:] = np.arange(16.0)
  args = (x, *args)
  before = [np.array(a, copy=True) for a in args]
  loaded = aotax.compile(fn, *args, out=tmp_path, name="update").load()
  want = jax.jit(fn)(*args)
  for _ in range(2):
    np.testing.assert_allclose(loaded(*args), want, rtol=1e-12, atol=1e-12)
    for a, b in zip(args, before, strict=True):
      np.testing.assert_array_equal(a, b)


@pytest.mark.xfail(strict=True, reason="IREE 3.12 slow_memcpy dispatch mishandles overlapping dynamic_update_slice")
def test_overlapping_dynamic_update(tmp_path):
  def update(x, i):
    return jax.lax.dynamic_update_slice(x, jax.lax.dynamic_slice(x, (i,), (6,)), (i + 2,))

  x = np.arange(16.0)
  loaded = aotax.compile(update, x, np.int32(3), out=tmp_path).load()
  got = loaded(x, np.int32(3))
  np.testing.assert_array_equal(x, np.arange(16.0))
  np.testing.assert_array_equal(got, jax.jit(update)(x, np.int32(3)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, False), (False, True), (True, True)])
def test_blas_matmul(tmp_path, monkeypatch, dtype, transpose_a, transpose_b):
  import sys

  rng = np.random.default_rng(4)
  a = rng.normal(size=(35, 37) if transpose_a else (37, 35)).astype(dtype)
  b = rng.normal(size=(33, 35) if transpose_b else (35, 33)).astype(dtype)

  def matmul(a, b):
    return (a.T if transpose_a else a) @ (b.T if transpose_b else b)

  if sys.platform == "darwin":
    generate = aotax.host

    def host(*args):
      text = generate(*args)
      assert "} else if (dispatch(" in text
      return "\n".join("    } else return 1;" if "} else if (dispatch(" in line else line for line in text.splitlines())

    monkeypatch.setattr(aotax, "host", host)
  check(matmul, a, b, tmp_path=tmp_path, rtol=1e-12 if dtype == np.float64 else 1e-5)
  host = (tmp_path / "iree" / "host.c").read_text()
  assert (f"cblas_{'d' if dtype == np.float64 else 's'}gemm(" in host) == (sys.platform == "darwin")


def test_blas_batch_and_slice(tmp_path):
  import sys

  rng = np.random.default_rng(5)
  a, b = rng.normal(size=(3, 35, 31)), rng.normal(size=(3, 29, 27))

  def matmul(a, b):
    return a[:, 1:34, 1:30] @ b

  check(matmul, a, b, tmp_path=tmp_path)
  host = (tmp_path / "iree" / "host.c").read_text()
  assert ("cblas_dgemm(" in host) == (sys.platform == "darwin")
  if sys.platform == "darwin":
    assert "batch < 3" in host


def test_fused_contraction_fallback(tmp_path):
  rng = np.random.default_rng(6)

  def matmul(a, b):
    return jnp.sin(a @ b)

  check(matmul, rng.normal(size=(23, 19)), rng.normal(size=(19, 17)), tmp_path=tmp_path)
  assert "cblas_" not in (tmp_path / "iree" / "host.c").read_text()


def test_blas_constant_and_subnormals(tmp_path):
  import sys

  weights = np.full((29, 27), 1e300)

  def matmul(x):
    return x @ weights

  x = np.full((33, 29), 1e-310)
  before = x * 1e300
  check(matmul, x, tmp_path=tmp_path)
  np.testing.assert_array_equal(x * 1e300, before)
  assert ("cblas_dgemm(" in (tmp_path / "iree" / "host.c").read_text()) == (sys.platform == "darwin")


def test_small_contraction_fallback(tmp_path):
  def matmul(a, b):
    return a @ b

  check(matmul, np.ones((6, 4)), np.ones((4, 4)), tmp_path=tmp_path)
  assert "cblas_" not in (tmp_path / "iree" / "host.c").read_text()


def test_exp_range_and_special_values(tmp_path):
  x = np.concatenate([np.linspace(-745, 710, 10001), np.linspace(-1e-14, 1e-14, 101), [0.0, -0.0, np.inf, -np.inf, np.nan]])
  loaded = aotax.compile(jnp.exp, x, out=tmp_path, name="exp_range").load()
  assert "stablehlo.exponential" not in (tmp_path / "iree/stablehlo.mlir").read_text()
  with np.errstate(over="ignore", under="ignore"):
    expected = np.exp(x)
  expected[expected < np.finfo(np.float64).tiny] = 0
  got = loaded(x)
  finite = np.isfinite(expected) & (expected != 0)
  np.testing.assert_array_equal(got[~finite], expected[~finite])
  assert np.all(np.abs(got[finite] - expected[finite]) <= 2 * np.spacing(expected[finite]))


def test_unused_middle_argument(tmp_path):
  def update(x, unused, i):
    return x.at[i].set(42.0)

  check(update, np.arange(4.0), np.full(3, 7.0), np.int64(2), tmp_path=tmp_path)
