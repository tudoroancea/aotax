"""Scalar leaves preserve absent tangents in small local programs; contractions stay arrays."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer, eval_jaxpr
from jax.extend import core

from aotax._sparse import ROUTING, Graph, routing


def obj(value):
  if isinstance(value, np.ndarray):
    return value
  out = np.empty((), dtype=object)
  out[()] = value
  return out


def scalarize(fn, *specs):
  closed = jax.make_jaxpr(fn)(*specs)
  graph = Graph(closed, -1)

  def scalar_fn(xs, *params):
    args = []
    for values, spec in zip((xs, *params), specs):
      array = np.empty(spec.shape, dtype=object)
      for i, index in enumerate(np.ndindex(spec.shape)):
        array[index] = values[i] if not args else values[index]
      args.append(array)
    cache = {}

    def read(n):
      if n in cache:
        return cache[n]
      if n.arg is not None:
        return args[n.arg]
      if n.value is not None:
        return np.asarray(n.value, dtype=object)
      a = [read(v) for v in n.ins]
      e, p = n.eq, n.eq.params
      name = e.primitive.name
      if name in ROUTING:
        out = np.concatenate([x.reshape(-1) for x in a])[routing(graph, n)[0]].reshape(n.shape)
      elif name == "reduce_sum":
        out = a[0].sum(axis=p["axes"])
      elif name == "dot_general":
        (lc, rc), (lb, rb) = p["dimension_numbers"]
        assert not lb and not rb
        out = np.tensordot(a[0], a[1], axes=(lc, rc))
      else:
        arrays = np.broadcast_arrays(*a)
        out = np.empty(arrays[0].shape, dtype=object)
        for idx in np.ndindex(out.shape):
          vals = [v[idx] for v in arrays]
          if not any(isinstance(v, Tracer) for v in vals):
            with jax.ensure_compile_time_eval():
              out[idx] = np.asarray(e.primitive.bind(*[np.asarray(v) for v in vals], **p)).item()
          else:
            out[idx] = Graph.custom(e)(*vals) if name == "custom_jvp_call" else e.primitive.bind(*vals, **p)
      cache[n] = obj(out)
      return cache[n]

    result = read(graph.output)
    return result.item() if result.ndim == 0 else tuple(result.reshape(-1))

  return scalar_fn


def differentiate(fn, specs, hessian, matrix, outputs):
  derivative: Callable[..., Any]
  width = specs[0].size
  mode = "hessian" if hessian else ("jacfwd" if width <= outputs else "jacrev")
  if matrix:
    from aotax.sparse import _asdex

    pattern = _asdex(fn, specs, 0, hessian)
    r, c = pattern.rows, pattern.cols
    active = np.unique(np.r_[r, c]) if hessian else np.arange(width)

    def reduced(x, *p):
      return fn(x if len(active) == width else jnp.zeros(width, dtype=x.dtype).at[active].set(x), *p)

    array_derivative = (jax.hessian if hessian else (jax.jacfwd if width <= outputs else jax.jacrev))(reduced)
    rr, cc = (np.searchsorted(active, r), np.searchsorted(active, c)) if hessian else (r, c)

    def values(x, *p):
      return array_derivative(x[active], *p)[rr, cc]

  else:
    sf = scalarize(fn, *specs)
    r, c = np.tril_indices(width) if hessian else np.indices((outputs, width)).reshape(2, -1)
    if hessian:
      derivative = jax.hessian(sf)
    elif width <= outputs:
      derivative = jax.jacfwd(sf)
    else:
      gradients = [jax.grad(lambda xs, *p, i=i: sf(xs, *p)[i]) for i in range(outputs)]
      derivative = lambda xs, *p: tuple(g(xs, *p) for g in gradients)

    def entries(x, *p):
      d = derivative(tuple(x[i] for i in range(width)), *p)
      return tuple(d[int(i)][int(j)] for i, j in zip(r, c))

    closed = jax.make_jaxpr(entries)(*specs)
    constants = dict(zip(closed.jaxpr.constvars, closed.consts))
    for e in closed.jaxpr.eqns:
      if all(isinstance(v, core.Literal) or v in constants for v in e.invars):
        args = [v.val if isinstance(v, core.Literal) else constants[v] for v in e.invars]
        with jax.ensure_compile_time_eval():
          value = e.primitive.bind(*args, **e.params)
        constants.update(zip(e.outvars, value if e.primitive.multiple_results else [value]))
    keep = np.asarray(
      [not np.all(np.asarray(v.val if isinstance(v, core.Literal) else constants.get(v, 1)) == 0) for v in closed.jaxpr.outvars], dtype=bool
    )
    r, c = r[keep], c[keep]
    positions = np.flatnonzero(keep)

    def values(x, *p):
      entries = eval_jaxpr(closed.jaxpr, closed.consts, x, *p)
      return jnp.stack([entries[i] for i in positions]) if len(positions) else jnp.zeros(0, dtype=x.dtype)

  return jax.make_jaxpr(values)(*specs), r, c, mode + ("; matrix products retained" if matrix else "; scalar leaves")
