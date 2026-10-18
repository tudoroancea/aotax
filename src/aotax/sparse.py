"""Sparse JAX derivatives with static, row-major matrix coordinates."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SparseDerivative:
  """A values function of the original arguments and its immutable matrix coordinates."""

  values: Callable[..., Any]
  rows: np.ndarray
  cols: np.ndarray
  shape: tuple[int, int]
  report: str


def sparse_hessian(f: Callable, *args: Any, argnums: int = 0, method: str = "auto") -> SparseDerivative:
  """Lower triangle of a scalar function's Hessian with respect to one real floating array.

  `args` supply arrays or ShapeDtypeStructs. Parameters remain runtime arguments of
  `values`; numerical samples never determine the structural pattern.
  """
  return _derivative(f, args, argnums, method, hessian=True)


def sparse_jacobian(f: Callable, *args: Any, argnums: int = 0, method: str = "auto") -> SparseDerivative:
  """Jacobian of one array output, flattening both input and output in row-major order."""
  return _derivative(f, args, argnums, method, hessian=False)


def _derivative(f, args, argnums, method, *, hessian):
  if method not in {"auto", "structured", "asdex"}:
    raise ValueError("method must be 'auto', 'structured' or 'asdex'")
  if not isinstance(argnums, int) or not -len(args) <= argnums < len(args):
    raise ValueError("argnums must select one array argument")
  argnums %= len(args)
  if any(not hasattr(a, "shape") or not hasattr(a, "dtype") for a in args):
    raise TypeError("args must be arrays or jax.ShapeDtypeStructs")
  specs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args)
  selected = specs[argnums]
  if not np.issubdtype(selected.dtype, np.floating):
    raise TypeError("the differentiated argument must be a real floating array")
  output = jax.eval_shape(f, *specs)
  if not isinstance(output, jax.ShapeDtypeStruct) or not np.issubdtype(output.dtype, np.floating):
    raise TypeError("the function must return one real floating array")
  if hessian and output.shape != ():
    raise ValueError("a Hessian requires a scalar output")

  def flat(*xs):
    xs = list(xs)
    xs[argnums] = xs[argnums].reshape(selected.shape)
    y = f(*xs)
    return y if hessian else y.reshape(-1)

  flat_specs = list(specs)
  flat_specs[argnums] = jax.ShapeDtypeStruct((selected.size,), selected.dtype)
  reason = ""
  if method != "asdex":
    from aotax._sparse import Unsupported, structured

    try:
      result = structured(flat, flat_specs, argnums, hessian)
    except Unsupported as error:
      reason = str(error)
      if method == "structured":
        raise ValueError(f"structured derivatives: {reason}; use method='asdex' or 'auto'") from error
    else:

      def values(*xs):
        xs = list(xs)
        xs[argnums] = jnp.asarray(xs[argnums]).reshape(-1)
        return result.values(*xs)

      return SparseDerivative(values, result.rows, result.cols, result.shape, result.report)
  result = _asdex(flat, flat_specs, argnums, hessian)

  def values(*xs):
    xs = list(xs)
    xs[argnums] = jnp.asarray(xs[argnums]).reshape(-1)
    return result.values(*xs)

  report = result.report if method == "asdex" else f"fallback: {reason}\n{result.report}"
  return SparseDerivative(values, result.rows, result.cols, result.shape, report)


def _indices(values):
  return np.frombuffer(np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


def _asdex(f, specs, argnums, hessian):
  try:
    import asdex
  except ModuleNotFoundError as error:
    if error.name != "asdex":
      raise
    raise ImportError("sparse derivatives require asdex; install aotax[sparse]") from error
  samples = tuple(np.zeros(s.shape, s.dtype) for s in specs)
  closed = jax.make_jaxpr(f)(*specs)
  operations = {eq.primitive.name for eq in _equations(closed)}
  custom = operations & {"custom_jvp_call", "custom_vjp_call"}
  _check_indices(closed, operations)
  if hessian and operations & {"cond", "while"}:
    raise ValueError(f"asdex: {min(operations & {'cond', 'while'})}: Hessian detection is unsupported")
  try:
    if custom:
      differentiated = jax.grad(f, argnums=argnums) if hessian else f
      output = jax.eval_shape(differentiated, *specs)

      def pullback(seed, *xs):
        return jax.vjp(differentiated, *xs)[1](seed)[argnums].reshape(-1)

      transposed = asdex.jacobian_sparsity(pullback, np.zeros(output.shape, output.dtype), *samples, argnums=0)
      pattern = asdex.SparsityPattern(transposed.cols, transposed.rows, transposed.shape[::-1], tuple(specs), argnums)
      if hessian:
        coloring = asdex.hessian_coloring_from_sparsity(pattern, symmetric=False)
      else:
        coloring = asdex.jacobian_coloring_from_sparsity(pattern, mode="rev" if "custom_vjp_call" in custom else None)
    elif hessian:
      coloring = asdex.hessian_coloring(f, *samples, argnums=argnums)
    else:
      coloring = asdex.jacobian_coloring(f, *samples, argnums=argnums, mode="fwd" if "while" in operations else None)
  except NotImplementedError as error:
    raise ValueError(f"asdex: unsupported derivative detection: {error}") from error
  compressed = (asdex.compressed_hessian_from_coloring if hessian else asdex.compressed_jacobian_from_coloring)(f, coloring)
  rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
  keep = np.flatnonzero(rows >= cols) if hessian else np.arange(rows.size)
  keep = keep[np.lexsort((cols[keep], rows[keep]))]
  # asdex 0.5.2's recovery pairs include star-coloring hub/spoke decisions.
  gather = np.asarray(coloring._gather_indices)[keep]

  def values(*xs):
    if (hessian or coloring.mode == "fwd") and any(s.dtype != specs[argnums].dtype for s in specs):

      def selected(x):
        return f(*xs[:argnums], x, *xs[argnums + 1 :])

      _, hvp = jax.linearize(jax.grad(selected) if hessian else selected, xs[argnums])
      data = jax.vmap(hvp)(coloring._device_seeds(specs[argnums].dtype))
    else:
      data = compressed(*xs)
    return data[gather[:, 0], gather[:, 1]]

  report = f"asdex {coloring.mode}: {coloring.num_colors} colors, {keep.size} entries; direct {'lower-triangle' if hessian else 'Jacobian'} recovery"
  if custom:
    report += "; JAX pullback structural detection preserves custom derivative rules"
    if hessian:
      report += "; nonsymmetric coloring does not assume custom rules have symmetric second derivatives"
  return SparseDerivative(values, _indices(rows[keep]), _indices(cols[keep]), coloring.sparsity.shape, report)


def _equations(value):
  stack = [value]
  while stack:
    item = stack.pop()
    if isinstance(item, (tuple, list)):
      stack.extend(item)
    elif hasattr(item, "eqns"):
      for eq in item.eqns:
        yield eq
        stack.extend(eq.params.values())
    elif hasattr(item, "jaxpr") and item.jaxpr is not item:
      stack.append(item.jaxpr)


def _check_indices(closed, operations):
  if not any(name == "gather" or name.startswith("scatter") for name in operations):
    return
  from aotax._sparse import Graph, Unsupported

  graph = Graph(closed, -1)
  pending, seen = [graph.output], set()
  while pending:
    n = pending.pop()
    if n in seen:
      continue
    seen.add(n)
    pending.extend(n.ins)
    if n.eq is None:
      continue
    name, p = n.eq.primitive.name, n.eq.params
    if name != "gather" and not name.startswith("scatter"):
      continue
    try:
      indices = graph.constant(n.ins[1])
    except Unsupported:
      continue
    dims = p["dimension_numbers"]
    if name == "gather":
      axes, sizes = dims.start_index_map, p["slice_sizes"]
    else:
      axes = dims.scatter_dims_to_operand_dims
      free = [d for d in range(len(n.shape)) if d not in (*dims.inserted_window_dims, *dims.operand_batching_dims)]
      sizes = [1] * len(n.shape)
      for d, u in zip(free, dims.update_window_dims):
        sizes[d] = n.ins[2].shape[u]
    bounds = np.array([n.ins[0].shape[d] - sizes[d] for d in axes])
    if np.any(indices < 0) or np.any(indices > bounds):
      raise ValueError(f"asdex: {name}: static out-of-bounds indices are unsupported by asdex 0.5.2 detection")
