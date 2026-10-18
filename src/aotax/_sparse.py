"""Local primal programs and static assembly for sparse JAX derivatives."""

from dataclasses import dataclass
from functools import cached_property

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
    source = n.ins[0]
    while source.eq is not None and source.eq.primitive.name in ROUTING - {"gather"}:
      source = source.ins[0]
    if source.dependent and source.arg is None:
      raise Unsupported("gather: selecting a shared computed map needs shared-map extraction")
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
      fn = lambda *a: n.eq.primitive.bind(*a, **p)
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

    def fn(a, b):
      result = n.eq.primitive.bind(a.reshape(ls), b.reshape(rs), **lp).reshape(-1)
      return result if np.array_equal(pick, np.arange(int(np.prod(localdims)))) else result[pick]

    return self.op(fn, (li, ri), "dot_general")

  def finish(self, outputs, hessian):
    selected = [ids for _, arg, ids in self.leaves if arg == self.graph.argnums]
    if not selected:
      return None
    inputs, _ = unique(np.concatenate(selected, axis=1))
    if inputs.shape[1] > 128:
      raise Unsupported(f"local input: width {inputs.shape[1]} exceeds 128; reduction/contraction mixes the candidate axis")
    if (
      hessian
      and inputs.shape[1] > 8
      and len(self.ops) > 128
      and not any(n.eq.primitive.name == "dot_general" and max(len(a.shape) for a in n.ins) >= 2 for n, _ in self.anchors)
    ):
      operation = self.anchors[-1][0].eq.primitive.name if self.anchors else "reduce_sum"
      raise Unsupported(f"{operation}: local Hessian width {inputs.shape[1]} with {len(self.ops)} operations needs nested sparse differentiation")
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
      result = jnp.stack([vals[i][0] for i in outputs])
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


def extract(f, specs, argnums, hessian):
  graph = Graph(jax.make_jaxpr(f)(*specs), argnums)
  roots = terms(graph)
  groups = []
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
    pending = [np.asarray(instances)]
    while pending:
      instances = pending.pop()
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
        pending.extend(instances[split.labels == k] for k in np.unique(split.labels))
        continue
      if piece is not None:
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
  n = specs[argnums].size
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
  from aotax._sparse_scalar import differentiate
  from aotax.sparse import SparseDerivative, _indices

  pieces, shape = extract(f, specs, argnums, hessian)
  evaluators, allrows, allcols, reports = [], [], [], []
  for fn, rows, inputs, params, matrix in pieces:
    local_specs = [jax.ShapeDtypeStruct((inputs.shape[1],), specs[argnums].dtype)]
    local_specs.extend(jax.ShapeDtypeStruct((ids.shape[1],), specs[arg].dtype) for arg, ids in params)
    closed, r, c, mode = differentiate(fn, local_specs, hessian, matrix, rows.shape[1])
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
      return jnp.zeros(0, dtype=specs[argnums].dtype)
    data = jnp.concatenate([*[evaluate(*xs) for evaluate in evaluators], jnp.zeros(1, dtype=specs[argnums].dtype)])
    parts = [data[gather[:, 0]] if gather.shape[1] == 1 else data[gather].sum(axis=1) for _, gather in buckets]
    return parts[0] if padded else jnp.concatenate(parts)[restore]

  report = "structured: grouped primal " + ("sums; local hessian (forward over reverse)" if hessian else "outputs; local Jacobians")
  reports.insert(
    0,
    "selection: at most 128 inputs, 512 array operations and backward-demand depth 128; non-matrix Hessians exceeding both width 8 and 128 operations use asdex",
  )
  return SparseDerivative(values, _indices(unique_keys // shape[1]), _indices(unique_keys % shape[1]), shape, "\n".join([report, *reports]))
