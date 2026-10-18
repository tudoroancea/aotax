"""Recognize complete dispatches that map exactly to Basic Linear Algebra Subprograms (BLAS)."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from iree.compiler import ir as mlir

ir: Any = mlir


class _Unsupported(Exception):
  pass


def _require(condition):
  if not condition:
    raise _Unsupported


def _walk(op):
  yield op
  for region in op.regions:
    for block in region.blocks:
      for child in block.operations:
        yield from _walk(child.operation)


@dataclass(frozen=True)
class _View:
  binding: int | str
  offset: str
  shape: tuple[int, ...]
  strides: tuple[int, ...]
  dtype: str
  data: bytes = b""

  @property
  def extent(self):
    return (1 + sum((n - 1) * s for n, s in zip(self.shape, self.strides, strict=True))) * (8 if self.dtype == "f64" else 4)

  def pointer(self, batch="0"):
    step = f" + {batch} * {self.strides[0]}" if len(self.shape) == 3 else ""
    base = self.binding if isinstance(self.binding, str) else f"ptrs[{self.binding}]"
    return f"(({'double' if self.dtype == 'f64' else 'float'}*)((unsigned char*){base} + {self.offset}){step})"


@dataclass(frozen=True)
class Contraction:
  a: _View
  b: _View
  c: _View
  accumulator: _View | None
  m: int
  n: int
  k: int
  batch: int
  transpose_a: bool
  transpose_b: bool

  def emit(self):
    a, b, c = self.a, self.b, self.c
    guards = [f"!overlap({v.pointer()}, {v.extent}, {c.pointer()}, {c.extent})" for v in (a, b)]
    if self.accumulator is not None and self.accumulator != c:
      v = self.accumulator
      guards.append(f"!overlap({v.pointer()}, {v.extent}, {c.pointer()}, {c.extent})")
    lines = []
    if self.batch > 1:
      lines.append(f"for (size_t batch = 0; batch < {self.batch}; ++batch) {{")
    batch = "batch" if self.batch > 1 else "0"
    if self.accumulator is not None and self.accumulator != c:
      v = self.accumulator
      lines.append(f"for (size_t row = 0; row < {self.m}; ++row)")
      lines.append(
        f"  memcpy({c.pointer(batch)} + row * {c.strides[-2]}, {v.pointer(batch)} + row * {v.strides[-2]}, {self.n} * sizeof(*{c.pointer()}));"
      )
    args = ["CblasRowMajor", *("CblasTrans" if t else "CblasNoTrans" for t in (self.transpose_a, self.transpose_b))]
    args += [str(self.m), str(self.n), str(self.k), "1", a.pointer(batch), str(a.strides[-2]), b.pointer(batch), str(b.strides[-2])]
    args += [str(int(self.accumulator is not None)), c.pointer(batch), str(c.strides[-2])]
    lines.append(f"cblas_{'d' if c.dtype == 'f64' else 's'}gemm({', '.join(args)});")
    if self.batch > 1:
      lines.append("}")
    return " && ".join(guards), lines


def _recognize(fn):
  _require(len(fn.regions[0].blocks) == 1 and not fn.regions[0].blocks[0].arguments)
  ops = [o.operation for o in fn.regions[0].blocks[0].operations]
  allowed = {
    "arith.constant",
    "arith.index_castui",
    "hal.interface.constant.load",
    "util.assume.int",
    "hal.interface.binding.subspan",
    "iree_tensor_ext.dispatch.tensor.load",
    "iree_tensor_ext.dispatch.tensor.store",
    "tensor.empty",
    "linalg.fill",
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.generic",
    "func.return",
  }
  _require(all(o.name in allowed for o in ops))
  contractions = [o for o in ops if o.name in {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}]
  stores = [o for o in ops if o.name == "iree_tensor_ext.dispatch.tensor.store"]
  _require(len(contractions) == len(stores) == 1)
  op, store = contractions[0], stores[0]
  _require(len(op.operands) == 3 and len(op.results) == 1 and store.operands[0] == op.results[0])
  block = op.regions[0].blocks[0]
  body = [o.operation for o in block.operations]
  _require(len(body) == 3 and [o.name for o in body] == ["arith.mulf", "arith.addf", "linalg.yield"])
  _require(set(body[0].operands) == set(block.arguments[:2]))
  _require(set(body[1].operands) == {body[0].results[0], block.arguments[2]} and list(body[2].operands) == list(body[1].results))
  maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
  _require(len(maps) == 3 and all(m.n_symbols == 0 and all(isinstance(e, ir.AffineDimExpr) for e in m.results) for m in maps))
  dims = [tuple(ir.AffineDimExpr(e).position for e in m.results) for m in maps]
  ad, bd, cd = dims
  _require(len(cd) in (2, 3) and cd == tuple(range(len(cd))))
  batch_dims, md, nd = cd[:-2], cd[-2], cd[-1]
  kd = len(cd)
  _require(all(m.n_dims == kd + 1 for m in maps))
  _require(ad in ((*batch_dims, md, kd), (*batch_dims, kd, md)) and bd in ((*batch_dims, kd, nd), (*batch_dims, nd, kd)))
  if op.name == "linalg.generic":
    _require([str(a) for a in op.attributes["iterator_types"]] == ["#linalg.iterator_type<parallel>"] * kd + ["#linalg.iterator_type<reduction>"])

  def expr(v):
    owner = v.owner.operation
    if owner.name == "arith.constant":
      _require(str(v.type) == "index" and int(owner.attributes["value"].value) >= 0)
      return str(int(owner.attributes["value"].value))
    if owner.name == "hal.interface.constant.load":
      return f"constants[{int(owner.attributes['ordinal'].value)}]"
    if owner.name == "arith.index_castui":
      _require(str(owner.operands[0].type) == "i32" and str(v.type) == "index")
      return expr(owner.operands[0])
    if owner.name == "util.assume.int":
      return expr(owner.operands[v.result_number])
    raise _Unsupported

  def view(access, target):
    span = target.owner.operation
    _require(span.name == "hal.interface.binding.subspan" and len(span.operands) <= 1)
    match = re.fullmatch(r"!iree_tensor_ext.dispatch.tensor<\w+:tensor<([\dx]+)(f32|f64)>>", str(target.type))
    if match is None:
      raise _Unsupported
    shape = tuple(int(n) for n in match[1].rstrip("x").split("x"))
    offsets, sizes, steps = [tuple(access.attributes[f"static_{key}"]) for key in ("offsets", "sizes", "strides")]
    _require(len(shape) == len(offsets) == len(sizes) == len(steps) and len(access.operands) == (2 if access.name.endswith("store") else 1))
    _require(
      all(n > 0 and off >= 0 and step > 0 and off + (n - 1) * step < dim for dim, off, n, step in zip(shape, offsets, sizes, steps, strict=True))
    )
    strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    offset = sum(o * s for o, s in zip(offsets, strides, strict=True)) * (8 if match[2] == "f64" else 4)
    base = expr(span.operands[0]) if span.operands else "0"
    strides = tuple(s * step for s, step in zip(strides, steps, strict=True))
    _require(strides[-1] == 1)
    return _View(int(span.attributes["binding"].value), f"({base} + {offset})", sizes, strides, match[2])

  def loaded(v, label):
    load = v.owner.operation
    if load.name == "arith.constant":
      t = ir.RankedTensorType(v.type)
      _require(str(t.element_type) in {"f32", "f64"} and not t.encoding and all(n > 0 for n in t.shape))
      data = np.asarray(ir.DenseElementsAttr(load.attributes["value"])).tobytes()
      return _View(f"blas_{label}", "0", tuple(t.shape), tuple(math.prod(t.shape[i + 1 :]) for i in range(t.rank)), str(t.element_type), data)
    _require(load.name == "iree_tensor_ext.dispatch.tensor.load")
    result = view(load, load.operands[0])
    t = ir.RankedTensorType(v.type)
    _require(tuple(t.shape) == result.shape and str(t.element_type) == result.dtype)
    return result

  a, b = [loaded(v, label) for v, label in zip(op.operands[:2], ("a", "b"), strict=True)]
  c = view(store, store.operands[1])
  t = ir.RankedTensorType(op.results[0].type)
  _require(tuple(t.shape) == c.shape and str(t.element_type) == c.dtype and a.dtype == b.dtype == c.dtype)
  sizes = {}
  for labels, v in zip(dims, (a, b, c), strict=True):
    _require(len(labels) == len(v.shape))
    for d, size in zip(labels, v.shape, strict=True):
      _require(d not in sizes or sizes[d] == size)
      sizes[d] = size
    _require(v.strides[-2] >= v.shape[-1] and (len(v.shape) == 2 or v.strides[0] >= v.shape[-2] * v.strides[-2]))
  _require(all(0 < n <= 2147483647 for n in (*sizes.values(), a.strides[-2], b.strides[-2], c.strides[-2])))
  accumulator = op.operands[2].owner.operation
  if accumulator.name == "linalg.fill":
    _require(accumulator.operands[0].owner.operation.name == "arith.constant")
    _require(float(accumulator.operands[0].owner.operation.attributes["value"].value) == 0)
    _require(accumulator.operands[1].owner.operation.name == "tensor.empty")
    acc = None
  else:
    acc = loaded(op.operands[2], "acc")
    _require(acc.shape == c.shape and acc.dtype == c.dtype)
  loads = {v.owner.operation for v in op.operands if v.owner.operation.name == "iree_tensor_ext.dispatch.tensor.load"}
  _require({o for o in ops if o.name == "iree_tensor_ext.dispatch.tensor.load"} == loads)
  _require(len([o for o in ops if o.name == "linalg.fill"]) == int(acc is None))
  return Contraction(a, b, c, acc, sizes[md], sizes[nd], sizes[kd], sizes.get(0, 1) if batch_dims else 1, ad[-1] == md, bd[-1] == kd)


def contractions(directory: Path) -> dict[str, Contraction]:
  result = {}
  with ir.Context():
    for path in sorted(directory.glob("*.mlir")):
      module = ir.Module.parse(path.read_text())
      exports = {o.attributes["sym_name"].value for o in _walk(module.operation) if o.name == "hal.executable.export"}
      for fn in _walk(module.operation):
        if fn.name != "func.func" or fn.attributes["sym_name"].value not in exports:
          continue
        try:
          contraction = _recognize(fn)
          threshold = 32768 if contraction.c.dtype == "f64" else 65536
          if 2 * contraction.m * contraction.n * contraction.k >= threshold:
            result[fn.attributes["sym_name"].value] = contraction
        except _Unsupported:
          pass
  return result
