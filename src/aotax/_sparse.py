"""Local primal programs and static assembly for sparse JAX derivatives."""

from dataclasses import dataclass, replace
from functools import cache, cached_property

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import eval_jaxpr
from jax.extend import core

ELEMENTWISE = {
  "abs",
  "acos",
  "acosh",
  "add",
  "add_any",
  "asin",
  "asinh",
  "atan",
  "atan2",
  "atanh",
  "cbrt",
  "ceil",
  "clamp",
  "complex",
  "conj",
  "convert_element_type",
  "cos",
  "cosh",
  "div",
  "eq",
  "exp",
  "exp2",
  "expm1",
  "floor",
  "ge",
  "gt",
  "imag",
  "integer_pow",
  "is_finite",
  "le",
  "log",
  "log1p",
  "log2",
  "logistic",
  "lt",
  "max",
  "min",
  "mul",
  "ne",
  "neg",
  "not",
  "or",
  "pow",
  "real",
  "reduce_precision",
  "rem",
  "round",
  "rsqrt",
  "select_n",
  "sign",
  "sin",
  "sinh",
  "sqrt",
  "square",
  "sub",
  "tan",
  "tanh",
  "one_minus_square",
  "and",
  "xor",
  "stop_gradient",
}
ROUTING = {"broadcast_in_dim", "slice", "squeeze", "reshape", "transpose", "stack", "concatenate", "pad", "split", "unstack", "rev", "tile", "gather"}


class Unsupported(ValueError):
  pass


class Split(Exception):
  def __init__(self, labels):
    self.labels = labels


def uniform(a):
  if len(a) > 1 and np.all(a == a[:1]):
    return np.broadcast_to(a[:1], a.shape)
  return a


def same(a):
  if len(a) > 1 and not np.all(a == a[:1]):
    _, labels = np.unique(a, axis=0, return_inverse=True)
    raise Split(labels)
  return a[0]


def unique(a):
  if a.shape[1] == 0:
    return a, np.zeros(0, dtype=int)
  repeated = len(a) > 1 and np.all(a == a[:1])
  work = a[:1] if repeated else a
  order = np.argsort(work, axis=1)
  s = np.take_along_axis(work, order, axis=1)
  mask = np.concatenate([np.ones((len(work), 1), bool), s[:, 1:] != s[:, :-1]], axis=1)
  counts = mask.sum(axis=1)
  if not np.all(counts == counts[0]):
    raise Split(counts)
  u = s[mask].reshape(len(work), int(counts[0]))
  rank = np.cumsum(mask, axis=1) - 1
  inv = np.take_along_axis(rank, np.argsort(order, axis=1), axis=1)
  return uniform(np.broadcast_to(u, (len(a), u.shape[1]))), same(inv)


@dataclass(eq=False)
class Node:
  shape: tuple
  dtype: object
  weak_type: bool = False
  arg: int | None = None
  value: object = None
  eq: object = None
  ins: tuple = ()
  output: int = 0
  dependent: bool = False

  @cached_property
  def size(self):
    return int(np.prod(self.shape))


class Graph:
  def __init__(self, closed, argnums):
    self.argnums = argnums
    self.routes = {}
    self.nodes = []
    self.inputs = [
      Node(v.aval.shape, v.aval.dtype, weak_type=v.aval.weak_type, arg=i, dependent=i == argnums) for i, v in enumerate(closed.jaxpr.invars)
    ]
    self.output = self.build(closed, self.inputs)[0]

  def build(self, closed, inputs):
    env = dict(zip(closed.jaxpr.invars, inputs))
    env.update(
      (v, Node(v.aval.shape, v.aval.dtype, weak_type=v.aval.weak_type, value=np.asarray(x))) for v, x in zip(closed.jaxpr.constvars, closed.consts)
    )

    def read(v):
      if isinstance(v, core.Literal):
        return Node(v.aval.shape, v.aval.dtype, weak_type=v.aval.weak_type, value=np.asarray(v.val))
      return env[v]

    for e in closed.jaxpr.eqns:
      if e.effects:
        raise Unsupported(f"{e.primitive.name}: effects are unsupported")
      ins = tuple(read(v) for v in e.invars)
      if e.primitive.name == "jit":
        outs = self.build(e.params["jaxpr"], ins)
      else:
        dependent = any(n.dependent for n in ins)
        outs = [
          Node(v.aval.shape, v.aval.dtype, weak_type=v.aval.weak_type, eq=e, ins=ins, output=i, dependent=dependent) for i, v in enumerate(e.outvars)
        ]
      self.nodes.extend(outs)
      env.update(zip(e.outvars, outs))
    return [read(v) for v in closed.jaxpr.outvars]

  def constant(self, n):
    if n.value is not None:
      return n.value
    if n.arg is not None:
      raise Unsupported("gather: runtime indices are unsupported")
    vals = [self.constant(x) for x in n.ins]
    with jax.ensure_compile_time_eval():
      n.value = np.asarray(
        n.eq.primitive.bind(*vals, **n.eq.params)[n.output] if n.eq.primitive.multiple_results else n.eq.primitive.bind(*vals, **n.eq.params)
      )
    return n.value

  @staticmethod
  def pointwise(closed):
    shape = closed.jaxpr.outvars[0].aval.shape
    for v in closed.jaxpr.invars:
      if v.aval.shape not in ((), shape):
        raise Unsupported("custom_jvp_call: rule changes array axes")
    for e in closed.jaxpr.eqns:
      if e.primitive.name == "custom_jvp_call":
        Graph.custom(e)
      elif e.primitive.name not in ELEMENTWISE:
        raise Unsupported(f"custom_jvp_call: rule operation {e.primitive.name} is not pointwise")
    return shape

  @staticmethod
  def custom(e):
    p = e.params
    if p["symbolic_zeros"]:
      raise Unsupported("custom_jvp_call: symbolic-zero rules are unsupported")
    if len(e.outvars) != 1:
      raise Unsupported("custom_jvp_call: multiple outputs are unsupported")
    if any(not np.issubdtype(v.aval.dtype, np.floating) for v in e.invars):
      raise Unsupported("custom_jvp_call: non-floating arguments are unsupported")
    if p["num_consts"]:
      raise Unsupported("custom_jvp_call: captured arguments are unsupported")
    primal = p["call_jaxpr"]
    if any(np.ndim(c) for c in primal.consts):
      raise Unsupported("custom_jvp_call: array constants are unsupported")
    Graph.pointwise(primal)
    jp, consts, zeros = p["jvp_jaxpr_fun"].call_wrapped(*([False] * len(e.invars)))
    if any(np.ndim(c) for c in consts):
      raise Unsupported("custom_jvp_call: array rule constants are unsupported")
    if any(zeros):
      raise Unsupported("custom_jvp_call: symbolic zero outputs are unsupported")
    # Recursive calls to the same custom rule in its primal are checked on demand.
    for op in jp.eqns:
      if op.primitive.name not in ELEMENTWISE | {"custom_jvp_call"}:
        raise Unsupported(f"custom_jvp_call: rule operation {op.primitive.name} is not pointwise")

    def replay(jaxpr, constants, args):
      env = dict(zip(jaxpr.constvars, constants))
      env.update(zip(jaxpr.invars, args))

      def read(v):
        return v.val if isinstance(v, core.Literal) else env[v]

      for op in jaxpr.eqns:
        a = [read(v) for v in op.invars]
        if op.primitive.name == "custom_jvp_call":
          out = [Graph.custom(op)(*a)]
        else:
          val = op.primitive.bind(*a, **op.params)
          out = val if op.primitive.multiple_results else [val]
        env.update(zip(op.outvars, out))
      return tuple(read(v) for v in jaxpr.outvars)

    @jax.custom_jvp
    def f(*xs):
      return replay(primal.jaxpr, primal.consts, xs)[0]

    @f.defjvp
    def jvp(xs, ts):
      ys = replay(jp, consts, (*xs, *ts))
      return ys[0], ys[1]

    return f


def routing(graph, n):
  if n in graph.routes:
    return graph.routes[n]
  name = n.eq.primitive.name
  p = n.eq.params
  sizes = np.array([a.size for a in n.ins])
  offsets = np.r_[0, np.cumsum(sizes)]
  a = [np.arange(lo, hi).reshape(v.shape) for v, lo, hi in zip(n.ins, offsets[:-1], offsets[1:])]
  if name == "reshape":
    out = (a[0].transpose(p["dimensions"]) if p.get("dimensions") else a[0]).reshape(n.shape)
  elif name == "squeeze":
    out = np.squeeze(a[0], axis=p["dimensions"])
  elif name == "transpose":
    out = a[0].transpose(p["permutation"])
  elif name == "slice":
    out = a[0][tuple(slice(lo, hi, st) for lo, hi, st in zip(p["start_indices"], p["limit_indices"], p["strides"] or (1,) * a[0].ndim))]
  elif name == "broadcast_in_dim":
    sh = [1] * len(n.shape)
    for i, d in enumerate(p["broadcast_dimensions"]):
      sh[d] = a[0].shape[i]
    out = np.broadcast_to(a[0].reshape(sh), n.shape)
  elif name == "stack":
    out = np.stack(a, axis=p["axis"])
  elif name == "concatenate":
    out = np.concatenate(a, axis=p["dimension"])
  elif name == "rev":
    out = np.flip(a[0], axis=p["dimensions"])
  elif name == "unstack":
    out = np.take(a[0], n.output, axis=p["axis"])
  elif name == "split":
    cuts = np.cumsum(p["sizes"])[:-1]
    out = np.split(a[0], cuts, axis=p["axis"])[n.output]
  elif name == "tile":
    out = np.tile(a[0], p["reps"])
  elif name == "pad":
    cfg = p["padding_config"]
    if any(lo < 0 or hi < 0 for lo, hi, _ in cfg):
      raise Unsupported("pad: negative padding")
    out = np.full(n.shape, offsets[1], dtype=int)
    out[tuple(slice(lo, lo + sz * (interior + 1), interior + 1) for (lo, hi, interior), sz in zip(cfg, n.ins[0].shape))] = a[0]
  elif name == "gather":
    index = graph.constant(n.ins[1])
    with jax.ensure_compile_time_eval():
      out = np.asarray(n.eq.primitive.bind(a[0], index, **dict(p, fill_value=-1)))
    if p["mode"] == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
      bounds = np.array([n.ins[0].shape[d] - p["slice_sizes"][d] for d in p["dimension_numbers"].start_index_map])
      if np.any(index < 0) or np.any(index > bounds):
        raise Unsupported("gather: out-of-bounds promise_in_bounds indices violate the indexing promise")
  else:
    raise Unsupported(f"{name}: unsupported index routing")
  if (name != "gather" and np.any(out < 0)) or np.any(out >= offsets[-1]):
    raise Unsupported(f"{name}: out-of-bounds fill is unsupported")
  graph.routes[n] = out.reshape(-1), offsets
  return graph.routes[n]


class Extract:
  def __init__(self, graph, count, merged=None):
    self.graph, self.count = graph, count
    self.merged = merged
    self.demands = {}
    self.anchors = []
    self.ops = []
    self.leaves = []
    self.cache = {}
    self.depth = 0

  def op(self, fn, args, name="gather"):
    i = len(self.ops)
    self.ops.append((fn, args))
    if i >= 512:
      raise Unsupported(f"{name}: local graph exceeds 512 array operations; nested extraction is needed")
    return i

  def demand(self, n, ids):
    if self.count * ids.shape[-1] > 2_000_000:
      raise Unsupported(f"{n.eq.primitive.name if n.eq else 'input'}: demand exceeds two million indices")
    ids = np.broadcast_to(ids, (self.count, ids.shape[-1]))
    if n.dependent and n.eq is not None and n.eq.primitive.name not in ROUTING:
      self.anchors.append((n, ids))
    self.demands.setdefault(n, []).append(ids)
    requested = ids
    if self.merged is not None and n.eq is not None and n in self.merged:
      demands = self.merged[n]
      if not n.dependent:
        shared = np.all(requested == requested[:1])
        demands = [d for d in demands if np.all(d == d[:1]) == shared]
      ids, _ = unique(np.concatenate([*demands, requested], axis=1))
    ids, _ = unique(ids)
    inv = same(np.stack([np.searchsorted(a, b) for a, b in zip(ids, requested)]))
    key = (id(n), ids[0].tobytes(), ids.strides[0], None if ids.strides[0] == 0 else ids.tobytes())
    if key not in self.cache:
      self.depth += 1
      if self.depth > 128:
        raise Unsupported(f"{n.eq.primitive.name if n.eq else 'input'}: local graph depth exceeds construction budget 128")
      try:
        self.cache[key] = self.build(n, ids)
      finally:
        self.depth -= 1
    result = self.cache[key]
    if len(inv) == ids.shape[1] and np.array_equal(inv, np.arange(len(inv))):
      return result
    return self.op(lambda x, ix=inv: x[ix], (result,))

  def build(self, n, ids):
    if n.value is not None:
      value = n.value.reshape(-1)[ids]
      same(value)
      v = value[0]
      return self.op(lambda v=v: jax.lax.convert_element_type_p.bind(v, new_dtype=n.dtype, weak_type=n.weak_type, sharding=None), (), "const")
    if n.arg is not None:
      self.leaves.append((len(self.ops), n.arg, ids))
      return self.op(None, (), "input")
    name = n.eq.primitive.name
    p = n.eq.params
    if name == "iota":
      n.value = self.graph.constant(n)
      return self.build(n, ids)
    if name in ROUTING:
      return self.route(n, ids)
    if name in ELEMENTWISE:
      inputs = []
      for a in n.ins:
        mapping = np.broadcast_to(np.arange(a.size).reshape(a.shape), n.shape).reshape(-1)
        inputs.append(self.demand(a, mapping[ids]))
      primitive = n.eq.primitive
      fn = lambda *a: primitive.bind(*a, **p)
      return self.op(fn, tuple(inputs), name)
    if name == "custom_jvp_call":
      f = Graph.custom(n.eq)
      inputs = []
      for a in n.ins:
        mapping = np.broadcast_to(np.arange(a.size).reshape(a.shape), n.shape).reshape(-1)
        inputs.append(self.demand(a, mapping[ids]))
      return self.op(f, tuple(inputs), name)
    if name == "reduce_sum":
      a = n.ins[0]
      axes = p["axes"]
      free = tuple(i for i in range(len(a.shape)) if i not in axes)
      mapping = np.arange(a.size).reshape(a.shape).transpose((*free, *axes)).reshape(n.size, -1)
      ix = mapping[ids].reshape(self.count, -1)
      val = self.demand(a, ix)
      width = ids.shape[1]
      red = mapping.shape[1]
      return self.op(lambda x: x.reshape(width, red).sum(axis=1), (val,), name)
    if name == "dot_general":
      return self.dot(n, ids)
    raise Unsupported(f"{name}: no backward array-demand rule")

  def route(self, n, ids):
    name = n.eq.primitive.name
    out, offsets = routing(self.graph, n)
    mapped = out[ids]
    sources = np.where(mapped < 0, -1, np.searchsorted(offsets[1:], mapped, side="right"))
    pattern = same(sources)
    parts = []
    order = []
    for i in np.unique(pattern):
      positions = np.flatnonzero(pattern == i)
      if i < 0:
        parts.append(self.op(lambda count=positions.size: jnp.full(count, n.eq.params["fill_value"], dtype=n.dtype), (), "gather fill"))
      else:
        parts.append(self.demand(n.ins[i], mapped[:, positions] - offsets[i]))
      order.extend(positions)
    perm = np.argsort(order)
    if len(parts) == 1 and np.array_equal(perm, np.arange(len(perm))):
      return parts[0]
    return self.op(lambda *xs: jnp.concatenate(xs)[perm], tuple(parts), name)

  def dot(self, n, ids):
    lhs, rhs = n.ins
    p = n.eq.params
    (lc, rc), (lb, rb) = p["dimension_numbers"]
    lf = tuple(i for i in range(len(lhs.shape)) if i not in (*lc, *lb))
    rf = tuple(i for i in range(len(rhs.shape)) if i not in (*rc, *rb))
    dims = [*(lhs.shape[i] for i in lb), *(lhs.shape[i] for i in lf), *(rhs.shape[i] for i in rf)]
    coords = np.stack(np.unravel_index(ids, tuple(dims)), axis=-1) if dims else np.zeros((*ids.shape, 0), int)
    axes = []
    for d in range(len(dims)):
      u, _ = unique(coords[:, :, d])
      axes.append(u)

    # A rectangular sub-contraction contains every demanded output. Contracted axes stay whole.
    def operand(a, batch, free, other):
      choices = []
      localshape = []
      for d, sz in enumerate(a.shape):
        if d in batch:
          choice = axes[batch.index(d)]
        elif d in free:
          choice = axes[len(lb) + other + free.index(d)]
        else:
          choice = np.broadcast_to(np.arange(sz), (self.count, sz))
        choices.append(choice)
        localshape.append(choice.shape[1])
      ix = np.zeros((self.count, *localshape), int)
      for d, (choice, sz) in enumerate(zip(choices, a.shape)):
        sh = [self.count] + [1] * len(localshape)
        sh[d + 1] = choice.shape[1]
        ix = ix * sz + choice.reshape(sh)
      return self.demand(a, ix.reshape(self.count, -1)), tuple(localshape)

    li, lshape = operand(lhs, lb, lf, 0)
    ri, rshape = operand(rhs, rb, rf, len(lf))
    localdims = tuple(x.shape[1] for x in axes)
    localcoords = []
    for d, u in enumerate(axes):
      localcoords.append(np.sum(u[:, :, None] < coords[:, None, :, d], axis=1))
    localids = np.ravel_multi_index(tuple(localcoords), localdims) if dims else np.zeros_like(ids)
    pick = same(localids)

    remove_l = {d for d in lf if lshape[d] == 1}
    remove_r = {d for d in rf if rshape[d] == 1}
    for l, r in zip(lb, rb):
      if lshape[l] == rshape[r] == 1:
        remove_l.add(l)
        remove_r.add(r)

    def adjusted(axes, removed):
      return tuple(d - sum(i < d for i in removed) for d in axes if d not in removed)

    lp = dict(p, dimension_numbers=((adjusted(lc, remove_l), adjusted(rc, remove_r)), (adjusted(lb, remove_l), adjusted(rb, remove_r))))
    ls = tuple(s for d, s in enumerate(lshape) if d not in remove_l)
    rs = tuple(s for d, s in enumerate(rshape) if d not in remove_r)

    primitive = n.eq.primitive

    def fn(a, b):
      result = primitive.bind(a.reshape(ls), b.reshape(rs), **lp).reshape(-1)
      return result if np.array_equal(pick, np.arange(int(np.prod(localdims)))) else result[pick]

    return self.op(fn, (li, ri), "dot_general")

  def finish(self, outputs, hessian, *, primal=False):
    selected = [ids for _, arg, ids in self.leaves if arg == self.graph.argnums]
    if not selected and not primal:
      return None
    inputs, _ = unique(np.concatenate(selected, axis=1)) if selected else (np.empty((self.count, 0), int), None)
    if inputs.shape[1] > 128:
      raise Unsupported(f"local input: width {inputs.shape[1]} exceeds 128; reduction/contraction mixes the candidate axis")
    params = []
    param_leaves = []
    lookups = {}
    for i, arg, ids in self.leaves:
      if arg == self.graph.argnums:
        relative = np.stack([np.searchsorted(x, y) for x, y in zip(inputs, ids)])
        lookups[i] = same(relative)
      else:
        params.append((arg, ids))
        param_leaves.append(i)

    def fn(x, *p):
      vals = []
      leaf = dict(zip(param_leaves, p))
      leaf.update((i, x if np.array_equal(ix, np.arange(inputs.shape[1])) else x[ix]) for i, ix in lookups.items())
      for i, (op, args) in enumerate(self.ops):
        vals.append(leaf[i] if op is None else op(*(vals[j] for j in args)))
      result = jnp.concatenate([vals[i] for i in outputs]) if primal else jnp.stack([vals[i][0] for i in outputs])
      return result.sum() if hessian else result

    return fn, inputs, params


def terms(graph):
  """Distribute only linear outer operations; nonlinear reductions remain intact."""
  result = []

  def walk(n, ids, rows, factors=(), sign=1):
    if not n.dependent or not len(ids):
      return
    name = n.eq.primitive.name if n.eq else "input"
    if name in ROUTING:
      mapping, offsets = routing(graph, n)
      mapped = mapping[ids]
      sources = np.searchsorted(offsets[1:], mapped, side="right")
      for i in np.unique(sources):
        keep = (sources == i) & (mapped >= 0)
        walk(n.ins[i], mapped[keep] - offsets[i], rows[keep], tuple((a, ix[keep], inverse) for a, ix, inverse in factors), sign)
    elif name in {"add", "sub", "neg"}:
      for i, a in enumerate(n.ins):
        mapping = np.broadcast_to(np.arange(a.size).reshape(a.shape), n.shape).reshape(-1)
        walk(a, mapping[ids], rows, factors, -sign if name == "neg" or (name == "sub" and i == 1) else sign)
    elif name == "reduce_sum":
      a = n.ins[0]
      axes = n.eq.params["axes"]
      free = tuple(i for i in range(len(a.shape)) if i not in axes)
      mapping = np.arange(a.size).reshape(a.shape).transpose((*free, *axes)).reshape(n.size, -1)
      width = mapping.shape[1]
      walk(a, mapping[ids].reshape(-1), np.repeat(rows, width), tuple((a, np.repeat(ix, width), inv) for a, ix, inv in factors), sign)
    elif n.size == 1 and name in {"mul", "div", "dot_general"} and (name != "dot_general" or not n.shape) and sum(a.dependent for a in n.ins) == 1:
      i = next(i for i, a in enumerate(n.ins) if a.dependent)
      if name == "div" and i == 1:
        result.append((n, ids, rows, factors, sign))
        return
      a, weight = n.ins[i], n.ins[1 - i]
      if name == "dot_general":
        (lc, rc), (lb, rb) = n.eq.params["dimension_numbers"]
        if lb or rb:
          raise Unsupported("dot_general: scalar weighted reduction has batch axes")
        contracted = (lc, rc)
        ix = np.arange(a.size).reshape(a.shape).transpose(contracted[i]).reshape(-1)
        wi = np.arange(weight.size).reshape(weight.shape).transpose(contracted[1 - i]).reshape(-1)
        walk(a, ix, np.repeat(rows, len(ix)), (*((p, np.repeat(pi, len(ix)), inv) for p, pi, inv in factors), (weight, wi, False)), sign)
      else:
        walk(a, ids, rows, (*factors, (weight, np.zeros(len(ids), int), name == "div")), sign)
    else:
      result.append((n, ids, rows, factors, sign))

  m = graph.output.size
  walk(graph.output, np.arange(m), np.arange(m))
  return result


def program(ex, term, indices):
  n, ids, _rows, factors, sign = term
  value = ex.demand(n, ids[indices, None])
  for a, ix, inverse in factors:
    weight = ex.demand(a, ix[indices, None])
    value = ex.op((lambda x, w: x / w) if inverse else (lambda x, w: x * w), (value, weight), "div" if inverse else "mul")
  return value if sign == 1 else ex.op(lambda x: -x, (value,), "neg")


def extract(graph, specs, hessian, *, linear=True, split_groups=False):
  roots = terms(graph) if linear else [(graph.output, np.arange(graph.output.size), np.arange(graph.output.size), (), 1)]
  groups = []
  neighborhoods = []
  pending = [(i, np.arange(len(term[1]))) for i, term in enumerate(roots)]
  while pending:
    i, indices = pending.pop()
    ex = Extract(graph, len(indices))
    try:
      output = program(ex, roots[i], indices)
      piece = ex.finish([output], hessian)
    except Split as split:
      pending.extend((i, indices[split.labels == k]) for k in np.unique(split.labels))
      if len(pending) + len(groups) > 256:
        raise Unsupported("local graph: more than 256 distinct index programs")
      continue
    matrix = any(n.eq.primitive.name == "dot_general" and max(len(a.shape) for a in n.ins) >= 2 for n, _ in ex.anchors)
    groups.append((i, indices, ex.anchors if piece is not None and (piece[1].shape[1] <= 8 or matrix) else []))
    neighborhoods.append(piece[1] if piece is not None and (not linear or (hessian and piece[1].shape[1] > 8)) else None)

  # Two demanded outputs share an instance when they demand the same nonlinear primal.
  offsets = np.r_[0, np.cumsum([len(indices) for _, indices, _ in groups])]
  parents = np.arange(offsets[-1])

  def find(i):
    while parents[i] != i:
      parents[i] = parents[parents[i]]
      i = parents[i]
    return i

  owners = {}
  for g, (_, indices, anchors) in enumerate(groups):
    neighborhood = neighborhoods[g]
    if neighborhood is not None:
      for k, inputs in enumerate(neighborhood):
        key = ("inputs", tuple(inputs))
        current = find(offsets[g] + k)
        if key in owners:
          parents[find(owners[key])] = current
        owners[key] = current
    for node, ids in anchors:
      for k, demanded in enumerate(ids):
        current = find(offsets[g] + k)
        for index in demanded:
          key = (node, int(index))
          if key in owners:
            other = find(owners[key])
            parents[other] = current
          else:
            owners[key] = current
  components = {}
  for g, (_, indices, _) in enumerate(groups):
    for k in range(len(indices)):
      components.setdefault(find(offsets[g] + k), []).append((g, k))
  buckets = {}
  for component in components.values():
    signature = tuple(g for g, _ in component)
    buckets.setdefault(signature, []).append([k for _, k in component])
  pieces = []
  for signature, instances in buckets.items():
    pending = [(signature, np.asarray(instances))]
    while pending:
      signature, instances = pending.pop()
      ex = Extract(graph, len(instances))
      outputs, rows = [], []
      try:
        for col, g in enumerate(signature):
          i, indices, _ = groups[g]
          selected = indices[instances[:, col]]
          outputs.append(program(ex, roots[i], selected))
          rows.append(roots[i][2][selected])
        ex = Extract(graph, len(instances), ex.demands)
        outputs = [program(ex, roots[groups[g][0]], groups[g][1][instances[:, col]]) for col, g in enumerate(signature)]
        piece = ex.finish(outputs, hessian)
      except Split as split:
        pending.extend((signature, instances[split.labels == k]) for k in np.unique(split.labels))
        continue
      except Unsupported:
        if not split_groups or len(signature) == 1:
          raise
        mid = len(signature) // 2
        pending.extend((signature[s], instances[:, s]) for s in (slice(None, mid), slice(mid, None)))
        continue
      if piece is not None:
        if len(pieces) >= 256:
          raise Unsupported("local graph: more than 256 distinct index programs")
        fn, inputs, params = piece
        pieces.append(
          (
            fn,
            np.stack(rows, axis=1),
            inputs,
            params,
            bool(any(n.eq.primitive.name == "dot_general" and max(len(a.shape) for a in n.ins) >= 2 for n, _ in ex.anchors)),
          )
        )
  n = specs[graph.argnums].size
  return pieces, (n if hessian else graph.output.size, n)


def input_map(ids):
  first = np.array2string(ids[0], threshold=16, max_line_width=120)
  if len(ids) == 1:
    return first
  stride = ids[1] - ids[0]
  if np.array_equal(ids, ids[0] + np.arange(len(ids))[:, None] * stride):
    return f"{first} + instance * {np.array2string(stride, threshold=16)}"
  return np.array2string(ids, threshold=32, max_line_width=120)


def structured(f, specs, argnums, hessian):
  graph = Graph(jax.make_jaxpr(f)(*specs), argnums)

  @cache
  def computed(n):
    return n.dependent and n.eq is not None and (n.eq.primitive.name not in ROUTING or any(computed(a) for a in n.ins))

  live, pending = set(), [graph.output]
  while pending:
    n = pending.pop()
    if n not in live:
      live.add(n)
      pending.extend(n.ins)
  sources = []
  for n in graph.nodes:
    if n in live and n.eq is not None and n.eq.primitive.name == "gather":
      source = n.ins[0]
      if computed(source) and source not in sources:
        sources.append(source)
  if sources:
    if len(sources) != 1:
      raise Unsupported("shared-map extraction: multiple computed gather sources")
    return compose(graph, specs, sources[0], hessian)
  try:
    pieces, shape = extract(graph, specs, hessian)
  except Unsupported:
    slices = {}
    for n in live:
      if n.eq is not None and n.eq.primitive.name == "slice" and computed(n.ins[0]):
        slices.setdefault(n.ins[0], []).append(routing(graph, n)[0])
    sources = [n for n, reads in slices.items() if any(np.intersect1d(a, b).size for i, a in enumerate(reads) for b in reads[:i])]
    if len(sources) == 1:
      return compose(graph, specs, sources[0], hessian)
    pieces, shape = extract(graph, specs, hessian, split_groups=True)
  return local_derivatives(pieces, shape, specs, argnums, hessian)


def local_derivatives(pieces, shape, specs, argnums, hessian, full=False):
  from aotax._sparse_scalar import differentiate

  evaluators, allrows, allcols, reports = [], [], [], []
  for fn, rows, inputs, params, matrix in pieces:
    local_specs = [jax.ShapeDtypeStruct((inputs.shape[1],), specs[argnums].dtype)]
    local_specs.extend(jax.ShapeDtypeStruct((ids.shape[1],), specs[arg].dtype) for arg, ids in params)
    closed, r, c, mode = differentiate(fn, local_specs, hessian, matrix, rows.shape[1], full=full)
    if not len(r):
      continue
    allrows.append((inputs[:, r] if hessian else rows[:, r]).reshape(-1))
    allcols.append(inputs[:, c].reshape(-1))
    live = set(closed.jaxpr.outvars)
    for eq in reversed(closed.jaxpr.eqns):
      if any(v in live for v in eq.outvars):
        live.update(v for v in eq.invars if not isinstance(v, core.Literal))
    affine = closed.jaxpr.invars[0] not in live

    def make(closed, inputs, params, affine):
      def local(*xs):
        return eval_jaxpr(closed.jaxpr, closed.consts, *xs)[0]

      def values(*xs):
        x = jnp.zeros(inputs.shape, dtype=specs[argnums].dtype) if affine else xs[argnums].reshape(-1)[inputs]
        ps = []
        axes: list[int | None] = [0]
        for arg, ids in params:
          shared = np.all(ids == ids[:1])
          ps.append(xs[arg].reshape(-1)[ids[0] if shared else ids])
          axes.append(None if shared else 0)
        return jax.vmap(local, in_axes=tuple(axes))(x, *ps).reshape(-1)

      return values

    evaluators.append(make(closed, inputs, params, affine))
    reports.append(
      f"{len(inputs)} local maps, width {inputs.shape[1]}, {rows.shape[1]} outputs/terms, input map {input_map(inputs)}; "
      f"{('input-independent hessian' if hessian else 'affine coefficients') if affine else mode}; parameters "
      + "; ".join(f"arg {arg}: {input_map(ids)}" for arg, ids in params)
    )
  report = "structured: grouped primal " + ("sums; local hessian (forward over reverse)" if hessian else "outputs; local Jacobians")
  reports.insert(
    0,
    "selection: at most 128 inputs, 512 array operations and backward-demand depth 128; "
    "scalar Hessian construction (input width * argument, result and operand sizes) at most 262144",
  )
  return assemble(evaluators, allrows, allcols, shape, specs[argnums].dtype, "\n".join([report, *reports]))


def compose(graph, specs, source, hessian):
  argnums = graph.argnums
  width = specs[argnums].size
  if not np.issubdtype(source.dtype, np.floating):
    raise Unsupported("shared map: outputs must be real floating arrays")
  if source.dtype != specs[argnums].dtype:
    raise Unsupported("shared map: mixed input/output floating dtypes are unsupported")
  saved = replace(source)
  source.arg, source.eq, source.ins = len(specs), None, ()
  pending = [(term, np.arange(len(term[1]))) for term in terms(graph)]
  needed = []
  while pending:
    term, indices = pending.pop()
    ex = Extract(graph, len(indices))
    try:
      program(ex, term, indices)
    except Split as split:
      pending.extend((term, indices[split.labels == k]) for k in np.unique(split.labels))
      if len(pending) > 256:
        raise Unsupported("shared map: more than 256 distinct index programs")
      continue
    needed.extend(ids.reshape(-1) for _, arg, ids in ex.leaves if arg == len(specs))
  source.__dict__.update(saved.__dict__)
  if not needed:
    pieces, shape = extract(graph, specs, hessian)
    return local_derivatives(pieces, shape, specs, argnums, hessian)
  needed = np.unique(np.concatenate(needed))
  output = graph.output
  spec = jax.ShapeDtypeStruct(source.shape, source.dtype)
  graph.output = graph.build(jax.make_jaxpr(lambda x: x.reshape(-1)[needed])(spec), [source])[0]
  maps, _ = extract(graph, specs, False, linear=False)
  graph.output = output
  if not maps:
    raise Unsupported("shared map: no used output depends on the selected input")
  active = np.unique(np.concatenate([needed[rows.reshape(-1)] for _, rows, _, _, _ in maps]))
  independent = np.setdiff1d(needed, active)
  independent_fn, independent_params = lambda x: x, []
  if len(independent):
    ex = Extract(graph, 1)
    independent_fn, _, independent_params = ex.finish([ex.demand(source, independent[None, :])], False, primal=True)

  def take(x, ids):
    flat = ids.reshape(-1)
    if len(flat) and np.array_equal(flat, np.arange(flat[0], flat[0] + len(flat))):
      return x.reshape(-1)[int(flat[0]) : int(flat[-1]) + 1].reshape(ids.shape)
    return x.reshape(-1)[ids]

  def primal(*xs):
    return independent_fn(jnp.zeros(0, specs[argnums].dtype), *[take(xs[arg], ids[0]) for arg, ids in independent_params])

  lifted_specs = list(specs)
  lifted_specs[argnums] = jax.ShapeDtypeStruct((width + len(active),), specs[argnums].dtype)
  lifted_specs.append(spec)
  selected = Node(lifted_specs[argnums].shape, specs[argnums].dtype, arg=argnums, dependent=True)
  original = graph.inputs[argnums]
  replacement = graph.build(jax.make_jaxpr(lambda x: x[:width].reshape(original.shape))(lifted_specs[argnums]), [selected])[0]
  original.__dict__.update(replacement.__dict__)
  passive = Node(source.shape, source.dtype, arg=len(specs))
  inactive = np.setdiff1d(np.arange(source.size), active)
  order = np.argsort(np.r_[active, inactive])
  replacement = graph.build(
    jax.make_jaxpr(lambda x, p: jnp.concatenate([x[width:], p.reshape(-1)[inactive]])[order].reshape(source.shape))(lifted_specs[argnums], spec),
    [selected, passive],
  )[0]
  source.__dict__.update(replacement.__dict__)
  graph.routes.clear()
  pieces, shape = extract(graph, lifted_specs, hessian)
  refined = []
  for fn, rows, ix, ps, _ in pieces:
    local_specs = [jax.ShapeDtypeStruct((ix.shape[1],), specs[argnums].dtype)]
    local_specs.extend(jax.ShapeDtypeStruct((ids.shape[1],), lifted_specs[arg].dtype) for arg, ids in ps)
    closed = jax.make_jaxpr(fn)(*local_specs)
    matrix = any(e.primitive.name == "dot_general" and any(len(getattr(v.aval, "shape", ())) >= 2 for v in e.invars) for e in closed.jaxpr.eqns)
    if not hessian:
      try:
        distinct, inverse = unique(rows)
      except Split:
        pass
      else:
        if distinct.shape[1] < rows.shape[1]:
          positions = [np.flatnonzero(inverse == i) for i in range(distinct.shape[1])]

          def grouped(x, *p, fn=fn, positions=positions):
            y = fn(x, *p)
            return jnp.stack([y[ix].sum() for ix in positions])

          fn, rows = grouped, distinct
    refined.append((fn, rows, ix, ps, matrix))
  pieces = refined
  outer = local_derivatives(pieces, shape, lifted_specs, argnums, hessian, full=hessian)
  gradient = None
  if hessian:
    gradients = [(lambda x, *p, fn=fn: jnp.atleast_1d(fn(x, *p)), np.zeros((len(ix), 1), int), ix, ps, matrix) for fn, _, ix, ps, matrix in pieces]
    gradient = local_derivatives(gradients, (1, shape[1]), lifted_specs, argnums, False)

  jets, prows, jrows, jcols, hrows, hcols = [], [], [np.arange(width)], [np.arange(width)], [], []
  jet_size = 0
  for fn, rows, inputs, params, _ in maps:
    rows = width + np.searchsorted(active, needed[rows])
    count, outputs, local_width = len(inputs), rows.shape[1], inputs.shape[1]
    prows.append(rows.reshape(-1) - width)
    jet_size += count * outputs * local_width ** (2 if hessian else 1)
    if jet_size > 2_000_000:
      raise Unsupported("shared map: local derivatives exceed two million entries")
    jrows.append(np.broadcast_to(rows[:, :, None], (count, outputs, local_width)).reshape(-1))
    jcols.append(np.broadcast_to(inputs[:, None, :], (count, outputs, local_width)).reshape(-1))
    r, c = np.tril_indices(local_width)
    if hessian:
      hrows.append(np.broadcast_to(rows[:, :, None], (count, outputs, len(r))).reshape(-1))
      hcols.append(np.broadcast_to((inputs[:, r] * width + inputs[:, c])[:, None, :], (count, outputs, len(r))).reshape(-1))

    def make(fn, inputs, params, r, c):
      jac = jax.jacfwd(lambda x, *p: (fn(x, *p), fn(x, *p)), has_aux=True)

      def first(x, *p):
        j, y = jac(x, *p)
        return j, (y, j)

      second = jax.jacfwd(first, has_aux=True)

      def evaluate(*xs):
        ps = []
        axes: list[int | None] = [0]
        for arg, ids in params:
          shared = np.all(ids == ids[:1])
          ps.append(take(xs[arg], ids[0] if shared else ids))
          axes.append(None if shared else 0)
        x = take(xs[argnums], inputs)
        if hessian:
          h, (y, j) = jax.vmap(second, in_axes=tuple(axes))(x, *ps)
          h = h[:, :, r, c].reshape(-1)
        else:
          j, y = jax.vmap(jac, in_axes=tuple(axes))(x, *ps)
          h = jnp.zeros(0, x.dtype)
        return y.reshape(-1), j.reshape(-1), h

      return evaluate

    jets.append(make(fn, inputs, params, r, c))
  jr, jc = np.concatenate(jrows), np.concatenate(jcols)
  incidence = np.split(np.argsort(jr, kind="stable"), np.cumsum(np.bincount(jr, minlength=width + len(active)))[:-1])
  products, rows, cols = [], [], []
  budget = 0
  for k, (r, c) in enumerate(zip(outer.rows, outer.cols)):
    right = incidence[c]
    budget += len(right) * (len(incidence[r]) if hessian else 1)
    if budget > 2_000_000:
      raise Unsupported("shared map: composition exceeds two million contributions")
    if hessian:
      left = incidence[r]
      a, b = np.repeat(left, len(right)), np.tile(right, len(left))
      keep = jc[a] >= jc[b]
      a, b = a[keep], b[keep]
      rows.extend(jc[a])
      products.extend(zip(np.full(len(a), k), a, b))
      cols.extend(jc[b])
    else:
      rows.extend(np.full(len(right), r))
      cols.extend(jc[right])
      products.extend(zip(np.full(len(right), k), right))
  products = np.asarray(products, dtype=int).reshape(-1, 3 if hessian else 2)
  hr, hc = (np.concatenate(hrows), np.concatenate(hcols)) if hessian else (np.zeros(0, int), np.zeros(0, int))
  gi = np.full(shape[1], -1, int)
  if gradient is not None:
    gi[gradient.cols] = np.arange(len(gradient.cols))
  keep = gi[hr] >= 0
  shape = (width if hessian else shape[0], width)
  map_primal = assemble([lambda data: data], prows, [np.zeros(len(r), int) for r in prows], (len(active), 1), specs[argnums].dtype, "")
  restore = np.full(saved.size, len(active) + len(independent), int)
  restore[active] = np.arange(len(active))
  restore[independent] = len(active) + np.arange(len(independent))

  def contributions(*xs):
    evaluated = [jet(*xs) for jet in jets]
    active_y = map_primal.values(jnp.concatenate([v[0] for v in evaluated]))
    y = jnp.concatenate([active_y, primal(*xs), jnp.zeros(1, specs[argnums].dtype)])[restore].reshape(saved.shape)
    lifted = list(xs)
    lifted[argnums] = jnp.concatenate([xs[argnums].reshape(-1), active_y])
    lifted.append(y)
    j = jnp.concatenate([jnp.ones(width, specs[argnums].dtype), *[v[1] for v in evaluated]])
    values = outer.values(*lifted)[products[:, 0]] * j[products[:, 1]]
    if gradient is not None:
      values = values * j[products[:, 2]]
      h = jnp.concatenate([v[2] for v in evaluated])
      values = jnp.concatenate([values, gradient.values(*lifted)[gi[hr[keep]]] * h[keep]])
    return values

  if hessian:
    rows.extend(hc[keep] // width)
    cols.extend(hc[keep] % width)
  report = (
    f"structured: retained shared map, {len(active)} live dependent outputs of {saved.size}; "
    f"{sum(len(ix) for _, _, ix, _, _ in maps)} instances, Jacobian and adjoint/Hessian composition; "
    f"{len(jr) - width} Jacobian entries, {len(hc) if hessian else 0} second derivatives, {len(rows)} contributions\n{outer.report}"
  )
  return assemble([contributions], [np.asarray(rows, int)], [np.asarray(cols, int)], shape, specs[argnums].dtype, report)


def assemble(evaluators, allrows, allcols, shape, dtype, report):
  from aotax.sparse import SparseDerivative, _indices

  rows = np.concatenate(allrows) if allrows else np.zeros(0, int)
  cols = np.concatenate(allcols) if allcols else np.zeros(0, int)
  keys = rows * shape[1] + cols
  unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
  width = counts.max(initial=1)
  order = np.argsort(inverse, kind="stable")
  starts = np.cumsum(counts) - counts
  buckets = []
  # Bounded padding keeps the common assembly unchanged; uneven counts use exact buckets.
  padded = len(unique_keys) * width <= 2 * len(keys)
  for count in [width] if padded else np.unique(counts):
    positions = np.arange(len(counts)) if padded else np.flatnonzero(counts == count)
    gather = np.full((len(positions), count), len(keys), dtype=np.int32)
    if padded:
      gather[inverse[order], np.arange(len(order)) - np.repeat(starts, counts)] = order
    else:
      gather[:] = order[starts[positions, None] + np.arange(count)]
    buckets.append((positions, gather))
  restore = None if padded else np.argsort(np.concatenate([positions for positions, _ in buckets]))

  def values(*xs):
    if not evaluators:
      return jnp.zeros(0, dtype=dtype)
    data = jnp.concatenate([*[evaluate(*xs) for evaluate in evaluators], jnp.zeros(1, dtype=dtype)])
    parts = [data[gather[:, 0]] if gather.shape[1] == 1 else data[gather].sum(axis=1) for _, gather in buckets]
    return parts[0] if padded else jnp.concatenate(parts)[restore]

  return SparseDerivative(values, _indices(unique_keys // shape[1]), _indices(unique_keys % shape[1]), shape, report)
