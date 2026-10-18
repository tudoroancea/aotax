"""Accurate elementary functions and optional dispatch approximations."""

import math
from typing import Any

import jax
import jax.numpy as jnp
from iree.compiler import ir as mlir
from jax.extend import core
from jax.interpreters import mlir as jax_mlir

from aotax.custom_calls import Call, Tensor, replace_call

ir: Any = mlir

FUNCTIONS = {
  "sine": "sin",
  "cosine": "cos",
  "exponential": "exp",
  "log": "log",
  "tanh": "tanh",
  "exponential_minus_one": "expm1",
  "log_plus_one": "log1p",
  "atan2": "atan2",
  "power": "pow",
  "cbrt": "cbrt",
  "tan": "tan",
}
COMPOSITES = {f"chlo.{name}": name for name in ("asin", "acos", "atan", "asinh", "acosh", "atanh", "sinh", "cosh", "tan", "erf", "erfc")}
DISPATCH_MATH = {"sin", "cos", "exp", "log", "tanh"}
PRESERVED = {name: core.Primitive(f"aotax_{name}") for name in ("erfc", "exp2", "log2", "log10")}


for _name, _primitive in PRESERVED.items():
  _primitive.def_abstract_eval(lambda x, **params: x)
  jax_mlir.register_lowering(_primitive, lambda ctx, x, function=_name, **params: jax.ffi.ffi_lowering(f"aotax.math.{function}")(ctx, x))


def preserve_math(fn, args):
  closed, shapes = jax.make_jaxpr(fn, return_shape=True)(*args)
  tree = jax.tree.structure(shapes)

  def rewrite(value):
    if isinstance(value, core.Jaxpr):
      return value.replace(eqns=[e.replace(primitive=PRESERVED.get(e.primitive.name, e.primitive), params=rewrite(e.params)) for e in value.eqns])
    if isinstance(value, dict):
      return {key: rewrite(v) for key, v in value.items()}
    if type(value) in (tuple, list):
      return type(value)(map(rewrite, value))
    return value

  evaluate = core.jaxpr_as_fun(rewrite(closed))
  return lambda *a: jax.tree.unflatten(tree, evaluate(*jax.tree.leaves(a)))


class MathCall(Call):
  @property
  def scratch(self):
    return 0

  def adapter(self, symbol):
    t, suffix = ("double", "") if self.inputs[0].dtype == "f64" else ("float", "f")
    args = [*(f"const void* p{i}" for i in range(len(self.inputs))), "void* r0", "void* scratch"]
    values = ", ".join(f"((const {t}*)p{i})[i]" for i in range(len(self.inputs)))
    return f"""static int {symbol}({", ".join(args)}) {{
  for (size_t i = 0; i < {math.prod(self.results[0].shape)}; ++i) (({t}*)r0)[i] = {self.target}{suffix}({values});
  return 0;
}}
"""


def accurate_math(text: str, device: str, approximate: bool = False) -> tuple[str, dict[str, Call]]:
  with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text)
    ops = []

    def collect(op):
      function = FUNCTIONS.get(op.name.removeprefix("stablehlo."))
      if op.name == "stablehlo.composite":
        function = COMPOSITES.get(op.attributes["name"].value)
      if op.name == "stablehlo.custom_call":
        target = op.attributes["call_target_name"].value
        if target in {f"aotax.math.{name}" for name in PRESERVED}:
          function = target.removeprefix("aotax.math.")
      if function and str(ir.RankedTensorType(op.results[0].type).element_type) in {"f32", "f64"}:
        ops.append((op, function))
      for region in op.regions:
        for block in region.blocks:
          for child in block.operations:
            collect(child.operation)

    collect(module.operation)
    registry = {}
    symbols = {o.attributes["sym_name"].value for o in module.body.operations if "sym_name" in o.attributes}
    for op, function in ops:
      t = ir.RankedTensorType(op.results[0].type)
      if approximate and not (device == "cpu" and function == "cbrt" and str(t.element_type) == "f64"):
        continue
      if device == "cpu" and function in DISPATCH_MATH:
        if str(t.element_type) == "f32":
          wide = ir.RankedTensorType.get(t.shape, ir.F64Type.get(), loc=op.location)
          convert = ir.Operation.create("stablehlo.convert", operands=list(op.operands), results=[wide], ip=ir.InsertionPoint(op), loc=op.location)
          op.operands[0] = convert.results[0]
          op.results[0].set_type(wide)
          narrow = ir.Operation.create("stablehlo.convert", operands=[op.results[0]], results=[t], ip=ir.InsertionPoint.after(op), loc=op.location)
          op.results[0].replace_all_uses_with(narrow.results[0])
          narrow.operands[0] = op.results[0]
        continue
      index = len(registry)
      while (symbol := f"aotax_math{index}") in symbols:
        index += 1
      symbols.add(symbol)

      def tensor(v):
        t = ir.RankedTensorType(v.type)
        return Tensor(tuple(t.shape), str(t.element_type))

      registry[symbol] = MathCall(function, tuple(map(tensor, op.operands)), tuple(map(tensor, op.results)), (), ())
      replace_call(module, op, symbol)
    return module.operation.get_asm(enable_debug_info=True), registry


def _exp(x):
  k = jax.lax.round(x * 1.4426950408889634, jax.lax.RoundingMethod.AWAY_FROM_ZERO)
  r = (x - k * 0.6931471803691238) - k * 1.9082149292705877e-10
  coefficients = [
    1.0,
    1.0,
    0.5000000000000019,
    0.1666666666666668,
    0.0416666666664881,
    0.008333333333319601,
    0.0013888888952314775,
    0.00019841269890047113,
    2.4801485482328494e-05,
    2.755724091857897e-06,
    2.763263963904103e-07,
    2.5110037605963777e-08,
  ]
  terms = [jnp.full_like(x, c) for c in coefficients]
  power = r
  while len(terms) > 1:
    terms = [terms[i] + power * terms[i + 1] if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
    power = power * power
  p = terms[0]
  n = k.astype(jnp.int64)
  a = n >> 1

  def scale(e):
    return jax.lax.bitcast_convert_type((e + 1023) << 52, jnp.float64)

  result = (p * scale(a)) * scale(n - a)
  result = jax.lax.select(x > 709.782712893384, jnp.full_like(x, jnp.inf), result)
  result = jax.lax.select(x < -708.3964185322641, jnp.zeros_like(x), result)
  return jax.lax.select(jax.lax.ne(x, x), x, result)


def expand_math(text: str, approximate: bool = False) -> str:
  with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text)
    ops = []
    shared = []
    has_dot = False

    def collect(op):
      nonlocal has_dot
      has_dot |= op.name in {"stablehlo.dot", "stablehlo.dot_general"}
      if (
        op.name == "stablehlo.exponential"
        and approximate
        and "result_accuracy" not in op.attributes
        and str(ir.RankedTensorType(op.results[0].type).element_type) == "f64"
      ):
        ops.append(op)
      elif (
        op.name in {"stablehlo.sine", "stablehlo.cosine", "stablehlo.tanh"}
        and str(ir.RankedTensorType(op.results[0].type).element_type) == "f64"
        and any(use.owner.name == "stablehlo.broadcast_in_dim" for use in op.results[0].uses)
      ):
        shared.append(op)
      for region in op.regions:
        for block in region.blocks:
          for child in block.operations:
            collect(child.operation)

    collect(module.operation)
    if not has_dot:
      for op in shared:
        barrier = ir.Operation.create(
          "stablehlo.optimization_barrier",
          operands=[op.results[0]],
          results=[op.results[0].type],
          ip=ir.InsertionPoint.after(op),
          loc=op.location,
        )
        op.results[0].replace_all_uses_with(barrier.results[0])
        barrier.operands[0] = op.results[0]
    helpers = {}
    for op in ops:
      t = ir.RankedTensorType(op.results[0].type)
      shape = tuple(t.shape)
      if shape not in helpers:
        helper = jax.jit(_exp).lower(jax.ShapeDtypeStruct(shape, jnp.float64)).as_text()
        helpers[shape] = ir.Module.parse(helper)
      block = helpers[shape].body.operations[0].regions[0].blocks[0]
      values = {block.arguments[0]: op.operands[0]}
      for child in block.operations:
        if child.operation.name == "func.return":
          op.results[0].replace_all_uses_with(values[child.operands[0]])
          break
        copy = child.operation.clone(ir.InsertionPoint(op))
        for i, operand in enumerate(child.operands):
          copy.operands[i] = values[operand]
        values.update(zip(child.results, copy.results, strict=True))
      op.erase()
    return module.operation.get_asm(enable_debug_info=True)
