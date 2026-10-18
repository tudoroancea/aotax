import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aotax

jax.config.update("jax_enable_x64", True)

FUNCTIONS = (
  "exp2",
  "log2",
  "log10",
  "sin",
  "cos",
  "exp",
  "log",
  "tanh",
  "expm1",
  "log1p",
  "arcsin",
  "arccos",
  "arctan",
  "arcsinh",
  "arccosh",
  "arctanh",
  "sinh",
  "cosh",
  "tan",
  "cbrt",
  "erf",
  "erfc",
  "arctan2",
  "power",
)


def inputs(name, dtype):
  info = np.finfo(dtype)
  small = np.nextafter(dtype(0), dtype(1))
  x = np.r_[np.linspace(-20, 20, 10001), np.linspace(-1e-12, 1e-12, 1001)]
  if name in {"sin", "cos", "tan", "arctan", "arcsinh", "cbrt", "arctan2"}:
    large = np.geomspace(1, float(info.max) / 2, 5001)
    x = np.r_[x, large, -large, np.linspace(-1e4, 1e4, 10001), -1e10, 1e10]
  elif name in {"exp", "exp2", "expm1", "sinh", "cosh"}:
    log = np.log2 if name == "exp2" else np.log
    x = np.r_[x, np.linspace(log(info.smallest_subnormal) - 2, log(info.max) + 2, 10001)]
  elif name in {"log", "log2", "log10", "log1p", "arccosh"}:
    x = np.r_[x, np.geomspace(float(small), float(info.max) / 2, 10001), np.linspace(0.9, 1.1, 10001)]
  elif name in {"arcsin", "arccos", "arctanh"}:
    x = np.r_[x, np.linspace(-1, 1, 10001), np.nextafter(dtype(1), dtype(0)), np.nextafter(dtype(-1), dtype(0))]
  x = np.r_[x, 0.0, -0.0, small, -small, info.tiny, -info.tiny, info.max, -info.max, np.inf, -np.inf, np.nan].astype(dtype)
  if name == "arctan2":
    return x, np.resize(np.array([0.0, -0.0, 1, -1, 1e-10, -1e10, np.inf, -np.inf, np.nan], dtype), x.shape)
  if name == "power":
    return x, np.resize(np.array([0, -0.0, 1, 2, 3, -1, -2, 0.5, 1.23, np.inf, -np.inf, np.nan], dtype), x.shape)
  return (x,)


def functions(name):
  if name in {"erf", "erfc"}:
    return getattr(jax.lax, name), lambda x: np.array([getattr(math, name)(float(v)) for v in x], dtype=x.dtype)
  return getattr(jnp, name), getattr(np, name)


def max_ulp(got, want):
  finite = np.isfinite(want) & np.isfinite(got)
  uint = np.uint32 if want.dtype == np.float32 else np.uint64
  sign = uint(1 << (31 if uint == np.uint32 else 63))

  def ordered(x):
    bits = x[finite].view(uint)
    return np.where(bits & sign, ~bits, bits | sign)

  a, b = ordered(got), ordered(want)
  return int(np.max(np.where(got[finite] == want[finite], uint(0), np.maximum(a, b) - np.minimum(a, b)), initial=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", FUNCTIONS)
def test_accurate_math(tmp_path, name, dtype):
  args = inputs(name, dtype)
  fn, reference = functions(name)
  lib = aotax.compile(fn, *args, out=tmp_path, name="math")
  with np.errstate(all="ignore"):
    want, got = reference(*args), lib.load()(*args)
  np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
  np.testing.assert_array_equal(np.isinf(got), np.isinf(want))
  np.testing.assert_array_equal(np.signbit(got[~np.isnan(want)]), np.signbit(want[~np.isnan(want)]))
  assert max_ulp(got, want) <= 2


@pytest.mark.parametrize("name", FUNCTIONS)
def test_approximate_math(tmp_path, name):
  fn, reference = functions(name)
  x = np.linspace(0.1, 0.9, 1001, dtype=np.float32)
  args = (x, x + 1) if name in {"power", "arctan2"} else (x + 1,) if name == "arccosh" else (x,)
  got = aotax.compile(fn, *args, out=tmp_path, name="math", approximate_math=True).load()(*args)
  np.testing.assert_allclose(got, reference(*args), rtol=5e-6, atol=5e-6)


def test_math_in_control_flow(tmp_path):
  def fn(x, n):
    return jax.lax.fori_loop(0, n, lambda i, v: jnp.log1p(v), x)

  x = np.linspace(-1e-8, 1e-8, 101)
  loaded = aotax.compile(fn, x, np.int32(3), out=tmp_path, name="math").load()
  for n in (0, 1, 3):
    want = x.copy()
    for _ in range(n):
      want = np.log1p(want)
    assert max_ulp(loaded(x, np.int32(n)), want) <= 4


@pytest.mark.parametrize("fn, reference", [(jnp.exp2, np.exp2), (jnp.log2, np.log2), (jax.lax.erfc, lambda x: np.array([math.erfc(v) for v in x]))])
def test_preserved_math_in_control_flow(tmp_path, fn, reference):
  def loop(x, n):
    return jax.lax.fori_loop(0, n, lambda i, v: fn(v), x)

  x = np.linspace(1.1, 1.2, 65)
  loaded = aotax.compile(loop, x, np.int32(2), out=tmp_path).load()
  assert max_ulp(loaded(x, np.int32(2)), reference(reference(x))) <= 4


def test_approximate_f64_cbrt(tmp_path):
  x = np.linspace(-100, 100, 1001)
  got = aotax.compile(jnp.cbrt, x, out=tmp_path, name="math", approximate_math=True).load()(x)
  assert max_ulp(got, np.cbrt(x)) <= 4


def test_recompile_reads_current_constants(tmp_path):
  weights = np.array([1.0, 2.0, 3.0])

  def fn(x):
    return x + weights

  x = np.array([10.0, 20.0, 30.0])
  aotax.compile(fn, x, out=tmp_path / "old")
  weights = np.array([4.0, 5.0, 6.0])
  np.testing.assert_array_equal(aotax.compile(fn, x, out=tmp_path / "new").load()(x), x + weights)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
def test_preserved_math_skips_other_dtypes(tmp_path, dtype):
  def fn(x):
    return jnp.real(jnp.exp2(x.astype(dtype)) + jnp.log2(x.astype(dtype))).astype(jnp.float32)

  x = np.linspace(0.5, 2, 9, dtype=np.float32)
  loaded = aotax.compile(fn, x, out=tmp_path).load()
  np.testing.assert_allclose(loaded(x), fn(x), rtol=2e-3)
