"""Preserve supported tensor calls through HAL and generate synchronous LAPACK adapters."""

import math
import re
import sys
from dataclasses import dataclass
from typing import Any

from iree.compiler import ir as mlir

ir: Any = mlir


@dataclass(frozen=True)
class Tensor:
  shape: tuple[int, ...]
  dtype: str

  @property
  def size(self):
    return math.prod(self.shape) * (8 if self.dtype == "f64" else 4)


@dataclass(frozen=True)
class Call:
  target: str
  inputs: tuple[Tensor, ...]
  results: tuple[Tensor, ...]
  config: tuple[tuple[str, int], ...]
  attributes: tuple[tuple[str, str], ...]

  @property
  def scratch(self):
    return math.prod(self.inputs[0].shape[-2:]) * (8 if self.inputs[0].dtype == "f64" else 4) if "trsm" not in self.target else 0

  def validate_hal(self, inputs, results):
    if [str(t) for t in inputs] != ["!hal.buffer_view"] * len(self.inputs) or [str(t) for t in results] != ["!hal.buffer_view"] * len(self.results):
      raise ValueError(f"aotax: {self.target}: lowered call signature differs from registered tensor signature")

  def adapter(self, symbol):
    t = "double" if self.inputs[0].dtype == "f64" else "float"
    prefix = "d" if t == "double" else "s"
    args = [*(f"const void* p{i}" for i in range(len(self.inputs))), *(f"void* r{i}" for i in range(len(self.results))), "void* scratch"]
    a = self.inputs[0].shape
    m, n = a[-2:]
    config = dict(self.config)
    if "trsm" in self.target:
      b = self.inputs[1].shape
      rows, cols = b[-2:]
      side = {76: "CblasLeft", 82: "CblasRight"}[config["side"]]
      uplo = {76: "CblasLower", 85: "CblasUpper"}[config["uplo"]]
      trans = {78: "CblasNoTrans", 84: "CblasTrans", 67: "CblasConjTrans"}[config["trans_x"]]
      diag = {78: "CblasNonUnit", 85: "CblasUnit"}[config["diag"]]
      body = f"""  for (size_t batch = 0; batch < {math.prod(b[:-2])}; ++batch) {{
    const {t}* a = (const {t}*)p0 + batch * {m * n};
    const {t}* b = (const {t}*)p1 + batch * {rows * cols};
    {t}* out = ({t}*)r0 + batch * {rows * cols};
    memcpy(out, b, {rows * cols} * sizeof({t}));
    cblas_{prefix}trsm(CblasRowMajor, {side}, {uplo}, {trans}, {diag}, {rows}, {cols}, 1, a, {n}, out, {cols});
  }}"""
    else:
      if "getrf" in self.target:
        factor = f"{prefix}getrf_(&m, &n, tmp, &lda, (int32_t*)r1 + batch * {min(m, n)}, &info);\n    ((int32_t*)r2)[batch] = info;"
      else:
        factor = f"char uplo = '{chr(config['uplo'])}';\n    {prefix}potrf_(&uplo, &n, tmp, &lda, &info);\n    ((int32_t*)r1)[batch] = info;"
      body = f"""  {t}* tmp = scratch;
  __LAPACK_int m = {m}, n = {n}, lda = {m}, info;
  for (size_t batch = 0; batch < {math.prod(a[:-2])}; ++batch) {{
    const {t}* a = (const {t}*)p0 + batch * {m * n};
    {t}* out = ({t}*)r0 + batch * {m * n};
    for (size_t i = 0; i < {m}; ++i) for (size_t j = 0; j < {n}; ++j) tmp[j * {m} + i] = a[i * {n} + j];
    {factor}
    for (size_t i = 0; i < {m}; ++i) for (size_t j = 0; j < {n}; ++j) out[i * {n} + j] = tmp[j * {m} + i];
  }}"""
    return f"static int {symbol}({', '.join(args)}) {{\n{body}\n  return 0;\n}}\n"


def _specialize(op):
  target = op.attributes["call_target_name"].value

  def require(condition, reason):
    if not condition:
      raise ValueError(f"aotax: custom call {target}: {reason}")

  match = re.fullmatch(r"lapack_([sd])(getrf|potrf|trsm)_ffi", target)
  if match is None:
    raise ValueError(f"aotax: custom call {target}: unsupported target")
  require(sys.platform == "darwin", "LAPACK adapters require macOS Accelerate; other platforms are untested")
  dtype, kind = ("f64" if match[1] == "d" else "f32"), match[2]
  require("has_side_effect" not in op.attributes or not op.attributes["has_side_effect"].value, "side effects are unsupported")
  require("api_version" not in op.attributes or int(op.attributes["api_version"].value) == 4, "expected typed FFI api_version 4")
  require("called_computations" not in op.attributes or not len(op.attributes["called_computations"]), "called computations are unsupported")

  def tensor(v):
    require(isinstance(v.type, ir.RankedTensorType), "requires ranked tensors")
    t = ir.RankedTensorType(v.type)
    require(not t.encoding and all(0 < n <= 2147483647 for n in t.shape), "requires static nonzero dimensions within LAPACK's 32-bit range")
    require(str(t.element_type) in {"f32", "f64", "i32"}, "unsupported dtype")
    return Tensor(tuple(t.shape), str(t.element_type))

  inputs, results = tuple(map(tensor, op.operands)), tuple(map(tensor, op.results))
  require(len(inputs) == (2 if kind == "trsm" else 1), "wrong number of inputs")
  a = inputs[0]
  require(a.dtype == dtype and len(a.shape) >= 2, "expected a matrix of the target dtype")
  batch, m, n = a.shape[:-2], *a.shape[-2:]
  backend = op.attributes["mhlo.backend_config"] if "mhlo.backend_config" in op.attributes else ir.DictAttr.get({})
  require(isinstance(backend, ir.DictAttr), "expected dictionary backend configuration")
  config = {}
  options = {} if kind == "getrf" else {"uplo": {76, 85}}
  if kind == "trsm":
    options |= {"side": {76, 82}, "trans_x": {78, 84, 67}, "diag": {78, 85}}
  require({attr.name for attr in backend} == set(options), "unsupported backend configuration")
  for key, allowed in options.items():
    value = backend[key]
    require(isinstance(value, ir.IntegerAttr) and int(value.value) in allowed, f"invalid {key}")
    config[key] = int(value.value)
  if kind == "getrf":
    expected = (a, Tensor((*batch, min(m, n)), "i32"), Tensor(batch, "i32"))
  elif kind == "potrf":
    require(m == n, "Cholesky requires square matrices")
    expected = (a, Tensor(batch, "i32"))
  else:
    b = inputs[1]
    require(m == n and b.dtype == dtype and len(b.shape) >= 2 and b.shape[:-2] == batch, "incompatible triangular solve matrices or batches")
    require(b.shape[-2 if config["side"] == 76 else -1] == n, "incompatible triangular solve dimensions")
    expected = (b,)
  require(results == expected, "result shapes or dtypes differ from the LAPACK contract")
  for key, tensors in (("operand_layouts", inputs), ("result_layouts", results)):
    require(key in op.attributes and len(op.attributes[key]) == len(tensors), f"missing or incompatible {key}")
    for layout, t in zip(op.attributes[key], tensors, strict=True):
      rank = len(t.shape)
      expected = (rank - 2, rank - 1, *reversed(range(rank - 2))) if t.dtype == dtype else tuple(reversed(range(rank)))
      require(isinstance(layout, ir.DenseIntElementsAttr) and tuple(layout) == expected, f"unsupported {key}: expected {expected}")
  attributes = tuple((key, str(op.attributes[key])) for key in op.attributes)
  return Call(target, inputs, results, tuple(sorted(config.items())), attributes)


def rewrite_custom_calls(text: str) -> tuple[str, dict[str, Call]]:
  with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text)
    calls = []

    def collect(op):
      if op.name == "stablehlo.custom_call":
        calls.append(op)
      for region in op.regions:
        for block in region.blocks:
          for child in block.operations:
            collect(child.operation)

    collect(module.operation)
    symbols = {o.attributes["sym_name"].value for o in module.body.operations if "sym_name" in o.attributes}
    registry = {}
    for op in calls:
      spec = _specialize(op)
      index = len(registry)
      while (symbol := f"aotax_cc{index}") in symbols:
        index += 1
      symbols.add(symbol)
      ins, outs = [str(v.type) for v in op.operands], [str(v.type) for v in op.results]
      declaration = ir.Module.parse(f"module {{ func.func private @{symbol}({', '.join(ins)}) -> ({', '.join(outs)}) }}")
      declaration.body.operations[0].operation.clone(ir.InsertionPoint(module.body))
      call = ir.Operation.create(
        "func.call",
        results=[v.type for v in op.results],
        operands=list(op.operands),
        attributes={"callee": ir.FlatSymbolRefAttr.get(symbol)},
        ip=ir.InsertionPoint(op),
        loc=op.location,
      )
      for old, new in zip(op.results, call.results, strict=True):
        old.replace_all_uses_with(new)
      registry[symbol] = spec
      op.erase()
    return module.operation.get_asm(enable_debug_info=True), registry
