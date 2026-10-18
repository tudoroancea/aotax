import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aotax
from aotax.cpu import fill_small_tiles

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("extent", [2, 3, 4, 6, 8, 16])
def test_dispatch_tiles(tmp_path, monkeypatch, extent):
  configurations = []

  def rewrite(text):
    updated = fill_small_tiles(text)
    configurations.extend(re.findall(r"vector_common_parallel = \[([^]]*)\]", t) for t in (text, updated))
    return updated

  monkeypatch.setattr(aotax, "fill_small_tiles", rewrite)
  x = np.arange(extent * 17, dtype=np.float64).reshape(extent, 17)
  loaded = aotax.compile(lambda x: x.T + 1, x, name="transpose", out=tmp_path).load()
  np.testing.assert_array_equal(loaded(x), x.T + 1)
  before, after = configurations
  assert before == ["2, 2"]
  assert after == [f"{extent if extent in (2, 4, 8) else 2}, 2"]
  assert "native_vector_size = 16" in (tmp_path / "iree/configured.mlir").read_text()


def test_tile_levels_and_budget():
  text = """
  #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  module {
    func.func @tiles(%x: tensor<8x8x17xf64>, %out: tensor<8x8x17xf64>) -> tensor<8x8x17xf64> {
      %0 = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"],
        lowering_config = #iree_cpu.lowering_config<distribution = [8, 8, 17],
          vector_common_parallel = [1, 1, 2], vector_reduction = [0, 1, 2], vector_inner_parallel = [1, 1, 64]>
      } ins(%x : tensor<8x8x17xf64>) outs(%out : tensor<8x8x17xf64>) {
      ^bb0(%a: f64, %b: f64):
        linalg.yield %a : f64
      } -> tensor<8x8x17xf64>
      return %0 : tensor<8x8x17xf64>
    }
  }
  """
  updated = fill_small_tiles(text)
  assert "distribution = [8, 8, 17]" in updated
  assert "vector_common_parallel = [1, 8, 2]" in updated
  assert "vector_reduction = [0, 8, 2]" in updated
  assert "vector_inner_parallel = [1, 1, 64]" in updated
  dynamic = fill_small_tiles(text.replace("8x8x17", "?x?x17"))
  assert "vector_common_parallel = [1, 1, 2]" in dynamic


def test_scalar_operands_and_scalable_tiles():
  text = """
  module {
    func.func @tiles(%s: f64, %out: tensor<8xf64>) -> tensor<8xf64> {
      %0 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"],
        lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [2]>
      } ins(%s : f64) outs(%out : tensor<8xf64>) {
      ^bb0(%a: f64, %b: f64):
        linalg.yield %a : f64
      } -> tensor<8xf64>
      return %0 : tensor<8xf64>
    }
  }
  """
  assert "vector_common_parallel = [8]" in fill_small_tiles(text)
  scalable = text.replace("vector_common_parallel = [2]", "vector_common_parallel = [[4], true]")
  assert "vector_common_parallel = [[4]," in fill_small_tiles(scalable)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("extent", [3, 4, 8])
def test_permutations(tmp_path, dtype, extent):
  def permute(x, y):
    a = jnp.transpose(x, (2, 0, 1))
    b = jnp.transpose(y, (1, 2, 0))
    return a + b * 0.5, jnp.transpose(a * b, (2, 1, 0)), jnp.transpose(x, (1, 0, 2))

  rng = np.random.default_rng(42)
  x = rng.normal(size=(extent, 4, 17)).astype(dtype)
  y = rng.normal(size=(4, 17, extent)).astype(dtype)
  loaded = aotax.compile(permute, x, y, out=tmp_path).load()
  for nonfinite in (False, True):
    if nonfinite:
      x.flat[::7] = np.resize(np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=dtype), x.flat[::7].size)
      y.flat[::11] = np.resize(np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], dtype=dtype), y.flat[::11].size)
    got, want = loaded(x, y), jax.jit(permute)(x, y)
    for actual, expected in zip(got, want, strict=True):
      np.testing.assert_allclose(actual, expected, rtol=1e-6 if dtype == np.float32 else 1e-12, equal_nan=True)
      zeros = (actual == 0) & (expected == 0)
      np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(np.asarray(expected)[zeros]))
