"""Fill small power-of-two axes in IREE's CPU vector tiles."""

import math
import re
from typing import Any

from iree.compiler import ir as mlir

ir: Any = mlir


def _walk(op):
  yield op
  for region in op.regions:
    for block in region.blocks:
      for child in block.operations:
        yield from _walk(child.operation)


def fill_small_tiles(text: str) -> str:
  with ir.Context():
    module = ir.Module.parse(text)
    for op in _walk(module.operation):
      if op.name != "linalg.generic" or "lowering_config" not in op.attributes:
        continue
      config = str(op.attributes["lowering_config"])
      if not config.startswith("#iree_cpu.lowering_config<"):
        continue
      dims = [0] * len(op.attributes["iterator_types"])
      maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
      if any(not m.is_projected_permutation for m in maps):
        continue
      for value, indexing in zip(op.operands, maps, strict=True):
        for extent, expr in zip(ir.ShapedType(value.type).shape, indexing.results, strict=True):
          if extent > 0:
            dims[ir.AffineDimExpr(expr).position] = extent

      def fill(match, dims=dims):
        tile = [int(n) for n in match[2].split(",") if n.strip()]
        if len(tile) == len(dims):
          for i in reversed(range(len(dims))):
            extent = dims[i]
            if 1 < extent <= 8 and extent & (extent - 1) == 0 and tile[i] > 0:
              candidate = [*tile[:i], extent, *tile[i + 1 :]]
              if math.prod(n or 1 for n in candidate) <= 32:
                tile = candidate
        return f"{match[1]} = [{', '.join(map(str, tile))}]"

      updated = re.sub(r"(vector_common_parallel|vector_reduction|vector_inner_parallel) = \[([^]]*)\]", fill, config)
      if updated != config:
        op.attributes["lowering_config"] = ir.Attribute.parse(updated)
    return module.operation.get_asm(enable_debug_info=True)
