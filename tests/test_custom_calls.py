import os
import subprocess
import sys
from itertools import product
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
import pytest

import aotax
from aotax.custom_calls import rewrite_custom_calls
from aotax.host import host

jax.config.update("jax_enable_x64", True)
lapack = pytest.mark.skipif(sys.platform != "darwin", reason="LAPACK adapters require Accelerate")


def check(fn, *args, tmp_path):
  loaded = aotax.compile(fn, *args, out=tmp_path, name="compute").load()
  tolerance = 2e-5 if any(np.asarray(a).dtype == np.float32 for a in args) else 1e-11
  for _ in range(2):
    before = [np.array(a) for a in args]
    jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w, rtol=tolerance, atol=tolerance), loaded(*args), jax.jit(fn)(*args))
    for a, b in zip(args, before, strict=True):
      np.testing.assert_array_equal(a, b)
  return loaded


@lapack
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [(), (2, 3)])
def test_solve(tmp_path, dtype, batch):
  rng = np.random.default_rng(12)
  a = (rng.normal(size=(*batch, 5, 5)) + np.eye(5)).astype(dtype)
  b = rng.normal(size=(*batch, 5, 2)).astype(dtype)

  def solve(a, b):
    return jnp.linalg.solve(a, b), a, b

  loaded = check(solve, a, b, tmp_path=tmp_path)
  for _ in range(2):
    a = (rng.normal(size=a.shape) + 2 * np.eye(5)).astype(dtype)
    np.testing.assert_allclose(loaded(a, b)[0], jax.jit(solve)(a, b)[0], rtol=2e-5 if dtype == np.float32 else 1e-11, atol=1e-5)


@lapack
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (2, 4, 4)])
def test_lu(tmp_path, dtype, shape):
  a = np.random.default_rng(13).normal(size=shape).astype(dtype)
  check(jax.lax.linalg.lu, a, tmp_path=tmp_path)


@lapack
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("left", [False, True])
def test_triangular_options(tmp_path, dtype, left):
  rng = np.random.default_rng(14)
  a = (rng.normal(size=(2, 4, 4)) + 4 * np.eye(4)).astype(dtype)
  b = rng.normal(size=(2, 4, 3) if left else (2, 3, 4)).astype(dtype)

  def triangular(a, b):
    return tuple(
      jax.lax.linalg.triangular_solve(a, b, left_side=left, lower=lower, transpose_a=trans, conjugate_a=conj, unit_diagonal=unit)
      for lower, trans, conj, unit in product([False, True], repeat=4)
    )

  check(triangular, a, b, tmp_path=tmp_path)


@lapack
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scipy_solvers_and_broadcasting(tmp_path, dtype):
  rng = np.random.default_rng(15)
  a = rng.normal(size=(2, 1, 4, 4)).astype(dtype)
  a = a @ np.swapaxes(a, -1, -2) + np.eye(4, dtype=dtype)
  b = rng.normal(size=(1, 3, 4, 2)).astype(dtype)

  def solvers(a, b):
    lower = jnp.linalg.cholesky(a)
    upper = jnp.linalg.cholesky(a, upper=True)
    lu, piv = jl.lu_factor(a)
    return (
      jnp.linalg.solve(a, b),
      jl.cho_solve((lower, True), b),
      jl.cho_solve((upper, False), b),
      *(jl.lu_solve((lu, piv), b, trans=t) for t in (0, 1, 2)),
      *(jl.solve_triangular(a, b, lower=l, trans=t, unit_diagonal=u) for l, t, u in product([False, True], [0, 1, 2], [False, True])),
    )

  check(solvers, a, b, tmp_path=tmp_path)


@lapack
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numerical_status(tmp_path, dtype):
  def status(a, b):
    return jnp.linalg.solve(a, b), jnp.linalg.cholesky(a), jax.lax.linalg.lu(a)

  a, b = np.ones((4, 4), dtype), np.ones(4, dtype)
  loaded = check(status, a, b, tmp_path=tmp_path)
  a = np.diag(np.array([-1, 2, 3, 4], dtype))
  jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w, equal_nan=True), loaded(a, b), jax.jit(status)(a, b))


@lapack
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_loops_and_live_old_results(tmp_path, dtype):
  a = np.array([[4, 1, 2], [0, 3, 1], [2, 1, 5]], dtype)
  b = np.array([1, 2, 3], dtype)

  def loops(a, b, count):
    def step(i, state):
      old, current = state
      result = jnp.linalg.solve(a + (i * 0.01) * jnp.eye(3, dtype=a.dtype), current + old * 0.1)
      return current, result

    fixed = jax.lax.fori_loop(0, 8, step, (b, b))
    _, old, current = jax.lax.while_loop(lambda s: s[0] < count, lambda s: (s[0] + 1, *step(s[0], s[1:])), (np.int32(0), *fixed))
    return old, current

  loaded = check(loops, a, b, np.int32(7), tmp_path=tmp_path)
  for n in (0, 1, 2, 12):
    jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w, rtol=2e-5, atol=1e-10), loaded(a, b, np.int32(n)), jax.jit(loops)(a, b, np.int32(n)))


@lapack
def test_newton_convergence(tmp_path):
  def residual(z, parameter):
    padded = jnp.pad(z, (1, 1))
    return (2 * z - padded[:-2] - padded[2:]) * (z.size + 1) ** 2 - parameter * jnp.exp(z)

  def newton(z, parameter):
    def step(state):
      i, z = state
      return i + 1, z - jnp.linalg.solve(jax.jacfwd(residual)(z, parameter), residual(z, parameter))

    return jax.lax.while_loop(lambda s: (s[0] < 20) & (jnp.max(jnp.abs(residual(s[1], parameter))) > 1e-10), step, (np.int32(0), z))

  z, parameters = np.zeros((4, 8)), np.linspace(0.5, 2.0, 4)
  loaded = check(jax.vmap(newton), z, parameters, tmp_path=tmp_path)
  np.testing.assert_array_equal(loaded(z, parameters)[0], [3, 3, 3, 4])


@lapack
@pytest.mark.parametrize("offset", [0, 1])
def test_caller_overlap_and_alignment(tmp_path, offset):
  def solve(a, b):
    return jnp.linalg.solve(a, b), b + 7

  a = np.array([[0.0, 2, 3], [4, 5, 6], [7, 8, 10]])
  b = np.arange(3.0)
  loaded = aotax._Loaded(aotax.compile(solve, a, b, out=tmp_path))
  raw = np.empty(512, np.uint8)
  start = -raw.ctypes.data % 64 + offset
  aa = np.ndarray((3, 3), np.float64, buffer=raw, offset=start)
  bb = np.ndarray((3,), np.float64, buffer=raw, offset=start + 128)
  out = np.ndarray((3,), np.float64, buffer=raw, offset=start + 8)
  for overlap_outputs in (False, True):
    aa[:], bb[:] = a, b
    second = out if overlap_outputs else bb
    assert loaded._call(loaded._context, aa.ctypes.data, bb.ctypes.data, out.ctypes.data, second.ctypes.data) == 0
    np.testing.assert_allclose(out, b + 7 if overlap_outputs else np.linalg.solve(a, b))
    if not overlap_outputs:
      np.testing.assert_allclose(bb, b + 7)


@lapack
@pytest.mark.parametrize("shape", [(0, 0), (2, 0, 0), (0, 3, 3)])
def test_empty_solve(tmp_path, shape):
  a, b = np.empty(shape), np.empty((*shape[:-1], 1))
  try:
    check(jnp.linalg.solve, a, b, tmp_path=tmp_path)
  except ValueError as error:
    assert "nonzero dimensions" in str(error)


def exported_solve():
  return jax.export.export(jax.jit(jnp.linalg.solve))(np.eye(3), np.ones(3)).mlir_module()


def test_unknown_target_and_platform(monkeypatch):
  with pytest.raises(ValueError, match="unsupported target"):
    rewrite_custom_calls(exported_solve().replace("lapack_dgetrf_ffi", "not_registered"))
  monkeypatch.setattr(sys, "platform", "linux")
  with pytest.raises(ValueError, match="macOS Accelerate"):
    rewrite_custom_calls(exported_solve())


@lapack
@pytest.mark.parametrize(
  "old,new,message",
  [
    ("uplo = 76", "uplo = 99", "invalid uplo"),
    ("diag = 85", "other = 85", "backend configuration"),
    ("lapack_dgetrf_ffi", "lapack_sgetrf_ffi", "target dtype"),
  ],
)
def test_bad_contract(old, new, message):
  with pytest.raises(ValueError, match=message):
    rewrite_custom_calls(exported_solve().replace(old, new))


@lapack
def test_hal_registry_validation():
  _, registry = rewrite_custom_calls(exported_solve())
  lib = SimpleNamespace()
  declaration = "module { util.func private @aotax_cc0(%a: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) }"
  with pytest.raises(ValueError, match="unregistered external declaration"):
    host(lib, declaration, [], "", {}, {}, [], {})
  with pytest.raises(ValueError, match="lowered call signature"):
    host(lib, declaration.replace("%a: !hal.buffer_view", "%a: i32"), [], "", {}, {}, [], registry)
  call = "module { util.func @main(%a: !hal.buffer_view) -> !hal.buffer_view { %0 = util.call @main(%a) : (!hal.buffer_view) -> !hal.buffer_view util.return %0 : !hal.buffer_view } }"
  lib.in_avals, lib.out_avals = (), ()
  with pytest.raises(ValueError, match="unregistered call main"):
    host(lib, call, [], "", {}, {}, [0], {})


@lapack
def test_adapter_error_restores_float_mode(tmp_path, monkeypatch):
  from aotax.custom_calls import Call

  adapter = Call.adapter
  monkeypatch.setattr(Call, "adapter", lambda self, symbol: adapter(self, symbol).replace("return 0;", "return 7;"))
  loaded = aotax.compile(jnp.linalg.solve, np.eye(3), np.ones(3), out=tmp_path, name="solve").load()
  tiny = np.array([1e-310])
  before = tiny * 1e300
  with pytest.raises(RuntimeError, match="solve_call failed"):
    loaded(np.eye(3), np.ones(3))
  np.testing.assert_array_equal(tiny * 1e300, before)


@lapack
def test_native_archive(tmp_path):
  def solve(a, b):
    return jax.lax.fori_loop(0, 3, lambda i, x: jnp.linalg.solve(a + i * jnp.eye(3), x), b)

  a = np.array([[0.0, 2, 3], [4, 5, 6], [7, 8, 10]])
  b = np.array([5.0, 15, 25])
  expected = np.asarray(jax.jit(solve)(a, b))
  aotax.compile(solve, a, b, out=tmp_path, name="solve")
  source = tmp_path / "main.c"
  source.write_text(f"""#include "solve.h"
#include <math.h>
int main(void) {{
  solve_context* c;
  double a[] = {{{", ".join(map(str, a.ravel()))}}}, b[] = {{5, 15, 25}}, x[3];
  double expected[] = {{{", ".join(map(str, expected))}}};
  if (solve_create(&c) || solve_call(c, a, b, x)) return 1;
  solve_destroy(c);
  for (int i = 0; i < 3; ++i) if (fabs(x[i] - expected[i]) > 1e-12) return 2;
  return 0;
}}
""")
  binary = tmp_path / "main"
  subprocess.run([os.environ.get("CC", "cc"), source, tmp_path / "libsolve.a", "-lm", "-framework", "Accelerate", "-o", binary], check=True)
  subprocess.run([binary], check=True)
  symbols = subprocess.run(["nm", "-g", tmp_path / "libsolve.a"], check=True, capture_output=True, text=True).stdout
  assert "aotax_cc" not in symbols
