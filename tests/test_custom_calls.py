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
@pytest.mark.parametrize("shape,bshape", [((0, 0), (0, 1)), ((2, 0, 0), (2, 0, 1)), ((0, 3, 3), (0, 3, 1)), ((3, 3), (3, 0))])
def test_empty_solve(tmp_path, shape, bshape):
  with pytest.raises(ValueError, match="nonzero dimensions"):
    aotax.compile(jnp.linalg.solve, np.empty(shape), np.empty(bshape), out=tmp_path, name="solve")


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
    ("dense<[0, 1]>", "dense<[1, 0]>", "unsupported operand_layouts"),
  ],
)
def test_bad_contract(old, new, message):
  with pytest.raises(ValueError, match=message):
    rewrite_custom_calls(exported_solve().replace(old, new))


@lapack
@pytest.mark.parametrize("old,new", [("[2, 3, 1, 0]", "[3, 2, 1, 0]"), ("[2, 1, 0]", "[1, 2, 0]"), ("[1, 0]", "[0, 1]")])
def test_result_layouts(old, new):
  a, b = np.broadcast_to(np.eye(4), (2, 3, 4, 4)), np.ones((2, 3, 4, 1))
  text = jax.export.export(jax.jit(jnp.linalg.solve))(a, b).mlir_module()
  lines = []
  for line in text.splitlines():
    before, marker, after = line.partition("result_layouts =")
    lines.append(before + marker + after.replace(old, new))
  with pytest.raises(ValueError, match="unsupported result_layouts"):
    rewrite_custom_calls("\n".join(lines))


@lapack
def test_fresh_result_keeps_inputs_bound(tmp_path, monkeypatch):
  from aotax.custom_calls import Call

  def aligned(shape):
    raw = np.empty(np.prod(shape) * 8 + 64, np.uint8)
    return np.ndarray(shape, np.float64, buffer=raw, offset=-raw.ctypes.data % 64)

  a, b = aligned((16, 16)), aligned((16, 16))
  a[:] = np.random.default_rng(16).normal(size=a.shape) + 16 * np.eye(16)
  b[:] = np.arange(256.0).reshape(b.shape)
  adapter = Call.adapter

  def checked(self, symbol):
    text = adapter(self, symbol)
    guard = f"  if ((uintptr_t)p0 != {a.ctypes.data} || (uintptr_t)p1 != {b.ctypes.data}) return 1;\n"
    return text.replace("{\n", "{\n" + guard, 1)

  monkeypatch.setattr(Call, "adapter", checked)

  def update(a, b):
    return jl.solve_triangular(a, b, lower=True).at[3].set(1.0)

  check(update, a, b, tmp_path=tmp_path)


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


def foreign_step(x, count, **kwargs):
  return jax.ffi.ffi_call("test_step", (jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct((), np.int64)), **kwargs)(x, count)


def build_foreign(lib, source):
  path = lib.header.parent / "foreign.c"
  path.write_text(f'#include "{lib.header.name}"\n' + source)
  obj = path.with_suffix(".o")
  subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-fPIC", "-c", path, "-o", obj], check=True)
  return obj


@pytest.mark.parametrize(
  "dtype,ctype,tag", [(np.float32, "float", "F32"), (np.float64, "double", "F64"), (np.int32, "int32_t", "I32"), (np.int64, "int64_t", "I64")]
)
def test_foreign_loops(tmp_path, dtype, ctype, tag):
  def loops(x, limit):
    def step(state):
      old, current, count = state
      result, count = foreign_step(current + old, count)
      return current, result, count

    fixed = jax.lax.fori_loop(0, 4, lambda i, state: step(state), (x, x, np.int64(0)))
    return jax.lax.while_loop(lambda state: state[2] < limit, step, fixed), x

  x = np.arange(6, dtype=dtype).reshape(2, 3)
  if dtype == np.int64:
    x += 1 << 40
  lib = aotax.compile(loops, x, np.int64(7), out=tmp_path, foreign={"test_step": aotax.Foreign("user_step", scratch=64)})
  obj = build_foreign(
    lib,
    f"""int user_step(const aotax_ffi_tensor* in, size_t ni, const aotax_ffi_tensor* out, size_t no, void* scratch) {{
  if (ni != 2 || no != 2 || !scratch || (uintptr_t)scratch % 64) return 1;
  if (in[0].dtype != AOTAX_FFI_{tag} || out[0].dtype != in[0].dtype || in[1].dtype != AOTAX_FFI_I64) return 2;
  if (in[1].rank || in[1].shape || out[1].rank || out[1].shape) return 3;
  if (in[0].data == out[0].data || in[1].data == out[1].data || out[0].data == out[1].data) return 4;
  size_t n = 1;
  for (size_t i = 0; i < in[0].rank; ++i) {{
    if (in[0].shape[i] != out[0].shape[i]) return 5;
    n *= in[0].shape[i];
  }}
  int64_t* count = scratch;
  *count = *(const int64_t*)in[1].data;
  for (size_t i = 0; i < n; ++i) (({ctype}*)out[0].data)[i] = ((const {ctype}*)in[0].data)[i] + *count;
  *(int64_t*)out[1].data = *count + 1;
  return 0;
}}
""",
  )
  loaded = lib.load(extra=[obj])
  for limit in (0, 4, 5, 9, 7):
    old, current = x.copy(), x.copy()
    for count in range(max(4, limit)):
      old, current = current, current + old + count
    result, original = loaded(x, np.int64(limit))
    for got, expected in zip(result, (old, current, max(4, limit)), strict=True):
      np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(original, x)
  assert "int user_step(const aotax_ffi_tensor*" in lib.header.read_text()
  symbols = subprocess.run(["nm", "-g", lib.archive], check=True, capture_output=True, text=True).stdout
  assert "U _user_step" in symbols if sys.platform == "darwin" else "U user_step" in symbols
  source = tmp_path / "main.c"
  source.write_text(f"""#include "loops.h"
int main(void) {{
  loops_context* c;
  {ctype} x[6] = {{0}}, old[6], current[6], original[6];
  int64_t limit = 7, count;
  if (loops_create(&c) || loops_call(c, x, &limit, old, current, &count, original)) return 1;
  loops_destroy(c);
  return count != 7 || current[0] != 46;
}}
""")
  binary = tmp_path / "main"
  subprocess.run([os.environ.get("CC", "cc"), source, lib.archive, obj, *aotax.SYSTEM_LIBS, "-o", binary], check=True)
  subprocess.run([binary], check=True)


def test_foreign_shapes_and_no_scratch(tmp_path):
  def fn(x):
    a, _ = foreign_step(x, np.int64(2))
    b, _ = foreign_step(x.reshape(-1), np.int64(3))
    return a, b

  x = np.arange(6.0).reshape(2, 3)
  lib = aotax.compile(fn, x, out=tmp_path, foreign={"test_step": "user_step"})
  obj = build_foreign(
    lib,
    """int user_step(const aotax_ffi_tensor* in, size_t ni, const aotax_ffi_tensor* out, size_t no, void* scratch) {
  if (scratch) return 1;
  size_t n = 1;
  for (size_t i = 0; i < in[0].rank; ++i) n *= in[0].shape[i];
  for (size_t i = 0; i < n; ++i) ((double*)out[0].data)[i] = ((const double*)in[0].data)[i] + *(const int64_t*)in[1].data;
  *(int64_t*)out[1].data = 0;
  return 0;
}
""",
  )
  for got, expected in zip(lib.load(extra=[obj])(x), (x + 2, x.ravel() + 3), strict=True):
    np.testing.assert_array_equal(got, expected)
  with pytest.raises(RuntimeError, match=r"load\(extra=.*") as error:
    lib.load()
  assert "user_step" in str(error.value)


@pytest.mark.parametrize(
  "options,attrs,message",
  [
    ({"has_side_effect": True}, {}, "side effects"),
    ({"input_output_aliases": {0: 0}}, {}, "aliases"),
    ({"input_layouts": [(1, 0), ()]}, {}, "operand_layouts"),
    ({"output_layouts": [(1, 0), ()]}, {}, "result_layouts"),
    ({}, {"scale": np.float64(2)}, "backend attributes"),
    ({}, {"opaque": "state"}, "backend attributes"),
    ({"custom_call_api_version": 2, "legacy_backend_config": "state"}, {}, "api_version"),
  ],
)
def test_foreign_rejects_options(tmp_path, options, attrs, message):
  def fn(x):
    return jax.ffi.ffi_call("test_step", (jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct((), np.int64)), **options)(
      x, np.int64(0), **attrs
    )

  with pytest.raises(ValueError, match=message):
    aotax.compile(fn, np.ones((2, 3)), out=tmp_path, foreign={"test_step": "user_step"})


def test_foreign_registry_and_tokens(tmp_path):
  with pytest.raises(ValueError, match="C identifier"):
    aotax.Foreign("not a symbol")
  with pytest.raises(ValueError, match="scratch"):
    aotax.Foreign("user_step", scratch=-1)
  with pytest.raises(ValueError, match="nonempty target"):
    aotax.compile(lambda x: x, np.ones(3), name="fn", out=tmp_path, foreign={"": "user_step"})
  with pytest.raises(ValueError, match="test_step"):
    aotax.compile(lambda x: foreign_step(x, np.int64(0)), np.ones(3), name="fn", out=tmp_path, foreign={"another": "user_step"})

  def fn(x):
    y, _ = jax.ffi.ffi_call("test_step", (jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ffi.abstract_token))(x)
    return y

  with pytest.raises(ValueError, match="tokens and ordered effects"):
    aotax.compile(fn, np.ones(3), out=tmp_path, foreign={"test_step": "user_step"})


def test_foreign_execution_error_and_header(tmp_path):
  def fn(x):
    return foreign_step(x, np.int64(0))

  lib = aotax.compile(fn, np.ones(3), out=tmp_path, foreign={"test_step": "user_step"})
  obj = build_foreign(
    lib,
    """int user_step(const aotax_ffi_tensor* in, size_t ni, const aotax_ffi_tensor* out, size_t no, void* scratch) {
  return 7;
}
""",
  )
  loaded = lib.load(extra=[obj])
  tiny = np.array([1e-310])
  before = tiny * 1e300
  with pytest.raises(RuntimeError, match="fn_call failed"):
    loaded(np.ones(3))
  np.testing.assert_array_equal(tiny * 1e300, before)
  source = tmp_path / "bad.c"
  source.write_text('#include "fn.h"\nint user_step(double* x) { return 0; }\n')
  result = subprocess.run([os.environ.get("CC", "cc"), "-c", source, "-o", tmp_path / "bad.o"], check=False, capture_output=True, text=True)
  assert result.returncode != 0
  assert "user_step" in result.stderr


@lapack
def test_foreign_and_lapack(tmp_path):
  def fn(a, b):
    return foreign_step(jnp.linalg.solve(a, b), np.int64(0))

  a, b = np.eye(3) * 2, np.arange(3.0)
  lib = aotax.compile(fn, a, b, out=tmp_path, foreign={"test_step": "user_step"})
  obj = build_foreign(
    lib,
    """int user_step(const aotax_ffi_tensor* in, size_t ni, const aotax_ffi_tensor* out, size_t no, void* scratch) {
  for (int i = 0; i < 3; ++i) ((double*)out[0].data)[i] = ((const double*)in[0].data)[i] * 2;
  *(int64_t*)out[1].data = 1;
  return 0;
}
""",
  )
  got, count = lib.load(extra=[obj])(a, b)
  np.testing.assert_array_equal(got, b)
  assert count == 1
