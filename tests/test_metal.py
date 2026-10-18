import ctypes
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aotax
from aotax.metal import validate

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def metal():
  if sys.platform != "darwin" or int(platform.mac_ver()[0].split(".")[0]) < 26:
    pytest.skip("Metal requires macOS 26 or later")
  framework = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
  framework.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
  device = framework.MTLCreateSystemDefaultDevice()
  objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
  objc.sel_registerName.argtypes = [ctypes.c_char_p]
  objc.sel_registerName.restype = ctypes.c_void_p
  objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
  objc.objc_msgSend.restype = ctypes.c_bool
  if not device or not objc.objc_msgSend(device, objc.sel_registerName(b"supportsFamily:"), 5002):
    pytest.skip("no Metal 4 device")


def check(fn, *args, tmp_path):
  library = aotax.compile(fn, *args, out=tmp_path, name="compute", device="metal")
  loaded = library.load()
  jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w, rtol=3e-5, atol=3e-5), loaded(*args), jax.jit(fn)(*args))
  assert "-framework Metal -framework Foundation" in library.header.read_text()
  return loaded


def test_elementwise_repeated_calls(tmp_path, metal):
  fn = lambda x: jnp.sin(x) * x + 2
  loaded = check(fn, np.arange(65, dtype=np.float32), tmp_path=tmp_path)
  for i in range(8):
    x = np.linspace(-2, 3, 65, dtype=np.float32) + i
    np.testing.assert_allclose(loaded(x), jax.jit(fn)(x), rtol=2e-5, atol=2e-5)


def test_matmul_and_mlp(tmp_path, metal):
  rng = np.random.default_rng(17)
  x, a, b, c = [rng.normal(size=shape).astype(np.float32) * 0.1 for shape in [(19, 9), (9, 32), (32, 32), (32, 2)]]
  check(lambda x, a: x @ a, x, a, tmp_path=tmp_path / "matmul")
  check(lambda x, a, b, c: jax.nn.sigmoid(jax.nn.sigmoid(x @ a) @ b) @ c, x, a, b, c, tmp_path=tmp_path / "mlp")


def test_pytrees_mixed_dtypes_and_unused_argument(tmp_path, metal):
  def step(state, unused, k):
    return {"x": state["x"] + state["v"], "v": state["v"] * 0.5}, (k * k + 1, jnp.sum(state["x"]))

  check(step, {"x": np.arange(4, dtype=np.float32), "v": np.ones(4, np.float32)}, np.ones(3, np.float32), np.int32(7), tmp_path=tmp_path)


def test_scan_while_and_branches(tmp_path, metal):
  def scan(x, xs):
    return jax.lax.scan(lambda carry, v: (carry * 0.5 + v, carry * v), x, xs)

  x, xs = np.ones(4, np.float32), np.arange(28, dtype=np.float32).reshape(7, 4)
  check(scan, x, xs, tmp_path=tmp_path / "scan")

  def loop(x, limit):
    return jax.lax.while_loop(lambda v: v[0] < limit, lambda v: (v[0] + 1, v[1] * 1.25 + v[0]), (np.int32(0), x))[1]

  loaded = check(loop, x, np.int32(9), tmp_path=tmp_path / "while")
  for limit in (0, 1, 3, 12):
    np.testing.assert_allclose(loaded(x, np.int32(limit)), loop(x, np.int32(limit)), rtol=1e-5)

  def branch(x, flag):
    return jax.lax.cond(flag > 0, jnp.sin, lambda a: a * a, x)

  loaded = check(branch, x, np.int32(1), tmp_path=tmp_path / "branch")
  np.testing.assert_array_equal(loaded(x, np.int32(-1)), x * x)


def test_constants_and_no_dispatches(tmp_path, metal):
  weights = np.arange(128, dtype=np.float32).reshape(16, 8) / 100
  check(lambda x: (weights @ x, x, x + 2, jnp.asarray(weights)), np.linspace(-1, 1, 8, dtype=np.float32), tmp_path=tmp_path / "constant")
  check(lambda x: (x, x, x), np.arange(4, dtype=np.float32), tmp_path=tmp_path / "identity")
  assert aotax.compile(lambda: (), out=tmp_path / "empty", name="empty", device="metal").load()() == ()


@pytest.mark.parametrize("kind", ["self_copy", "loop", "zero_fill", "nonzero_fill"])
def test_updates_preserve_inputs(tmp_path, metal, kind):
  functions = {
    "self_copy": lambda x: x.at[2:10].set(x[:8]),
    "loop": lambda x: jax.lax.fori_loop(0, 5, lambda i, v: v.at[1:].set(v[:-1]), x),
    "zero_fill": lambda x: x.at[2:10].set(0),
    "nonzero_fill": lambda x: x.at[2:10].set(3.25),
  }
  x = np.arange(16, dtype=np.float32)
  loaded = check(functions[kind], x, tmp_path=tmp_path)
  for _ in range(3):
    np.testing.assert_allclose(loaded(x), jax.jit(functions[kind])(x))
    np.testing.assert_array_equal(x, np.arange(16, dtype=np.float32))


def test_two_contexts(tmp_path, metal):
  fn = lambda x: jnp.sin(x) * x
  library = aotax.compile(fn, np.zeros(32, np.float32), out=tmp_path, name="compute", device="metal")
  contexts = [library.load(), library.load()]
  xs = [np.arange(32, dtype=np.float32) + i for i in range(8)]
  with ThreadPoolExecutor(2) as pool:
    results = list(pool.map(lambda context: [context(x) for x in xs], contexts))
  for outputs in results:
    for x, got in zip(xs, outputs, strict=True):
      np.testing.assert_allclose(got, fn(x), rtol=2e-5, atol=2e-5)


def test_create_reports_msl_errors(tmp_path, metal, monkeypatch, capfd):
  from aotax import metal as backend

  pipelines = backend._pipelines

  def invalid(binary):
    result = pipelines(binary)
    result[0]["source"] = "invalid Metal source"
    return result

  monkeypatch.setattr(backend, "_pipelines", invalid)
  library = aotax.compile(lambda x: x + 1, np.ones(4, np.float32), out=tmp_path, name="broken", device="metal")
  with pytest.raises(RuntimeError, match="broken_create failed"):
    library.load()
  assert "error:" in capfd.readouterr().err


@pytest.mark.parametrize("offset", [0, 1])
def test_staging_overlap_and_native_archive(tmp_path, metal, offset):
  loaded = check(lambda x: (x * 3, x + 7), np.arange(8, dtype=np.float32), tmp_path=tmp_path)
  raw = np.empty(128, np.uint8)
  x = np.ndarray((8,), np.float32, buffer=raw, offset=offset)
  x[:] = np.arange(8, dtype=np.float32)
  out = np.empty_like(x)
  assert loaded._call(loaded._context, x.ctypes.data, out.ctypes.data, x.ctypes.data) == 0
  np.testing.assert_array_equal(out, np.arange(8, dtype=np.float32) * 3)
  np.testing.assert_array_equal(x, np.arange(8, dtype=np.float32) + 7)
  assert loaded._call(loaded._context, x.ctypes.data, out.ctypes.data, out.ctypes.data) == 0
  np.testing.assert_array_equal(out, x + 7)
  source = tmp_path / "main.c"
  source.write_text(
    '#include "compute.h"\nint main(void) { compute_context* c; float x[8] = {0}, a[8], b[8];\n'
    "if (compute_create(&c) || compute_call(c, x, a, b)) return 1; compute_destroy(c); return b[0] != 7; }\n"
  )
  binary = tmp_path / "main"
  subprocess.run(["cc", source, tmp_path / "libcompute.a", "-framework", "Metal", "-framework", "Foundation", "-o", binary], check=True)
  subprocess.run([binary], check=True)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_rejects_64_bit_inputs_and_outputs(tmp_path, dtype):
  if sys.platform != "darwin":
    pytest.skip("Metal compilation requires macOS")
  with pytest.raises(TypeError, match="Metal supports float32 and int32 only"):
    aotax.compile(lambda x: x, np.ones(4, dtype), out=tmp_path, name="reject", device="metal")
  with jax.enable_x64(False), pytest.raises(TypeError, match="Metal supports float32 and int32 only"):
    aotax.compile(lambda x: x, np.ones(4, dtype), out=tmp_path, name="reject", device="metal")
  with pytest.raises(TypeError, match="Metal supports float32 and int32 only"):
    aotax.compile(lambda x: x.astype(dtype), np.ones(4, np.float32), out=tmp_path, name="reject", device="metal")


def test_rejects_float64_intermediates():
  text = jax.export.export(jax.jit(lambda x: (x.astype(np.float64) ** 2).astype(np.float32)))(np.ones(4, np.float32)).mlir_module()
  with pytest.raises(TypeError, match="float64 intermediates"):
    validate(text)


def test_internal_int64_is_not_demoted(tmp_path, metal):
  check(lambda x: (x.astype(np.int64) ** 2).astype(np.float32), np.array([70000, -80000], np.int32), tmp_path=tmp_path)


def test_rejects_custom_calls_and_unknown_device(tmp_path):
  text = jax.export.export(jax.jit(jnp.linalg.solve))(np.eye(3, dtype=np.float32), np.ones(3, np.float32)).mlir_module()
  with pytest.raises(ValueError, match="Metal does not support custom calls"):
    validate(text)
  with pytest.raises(ValueError, match="unsupported device"):
    aotax.compile(lambda x: x, np.ones(3), out=tmp_path, device="gpu")
