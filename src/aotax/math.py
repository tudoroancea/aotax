"""Prepare f64 math for dispatch vectorization."""

from typing import Any

import jax
import jax.numpy as jnp
from iree.compiler import ir as mlir

ir: Any = mlir


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


def expand_math(text: str) -> str:
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
