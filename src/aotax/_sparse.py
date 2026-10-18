"""Backward array demand extraction, without scalar expansion."""

import itertools
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
  arg: int | None = None
  value: object = None
  eq: object = None
  ins: tuple = ()
  output: int = 0
  dependent: bool = False
  number: int = -1

  @cached_property
  def size(self):
    return int(np.prod(self.shape))


class Graph:
  def __init__(self, closed, argnums):
    self.argnums = argnums
    self.counter = 0
    self.scatter_maps = {}
    self.inputs = [Node(v.aval.shape, v.aval.dtype, arg=i, dependent=i == argnums) for i, v in enumerate(closed.jaxpr.invars)]
    self.output = self.build(closed, self.inputs)[0]

  def build(self, closed, inputs):
    env = dict(zip(closed.jaxpr.invars, inputs))
    env.update((v, Node(v.aval.shape, v.aval.dtype, value=np.asarray(x))) for v, x in zip(closed.jaxpr.constvars, closed.consts))

    def read(v):
      if isinstance(v, core.Literal):
        return Node(v.aval.shape, v.aval.dtype, value=np.asarray(v.val))
      return env[v]

    for e in closed.jaxpr.eqns:
      if e.effects:
        raise Unsupported(f"{e.primitive.name}: effects are unsupported")
      ins = tuple(read(v) for v in e.invars)
      if e.primitive.name == "jit":
        outs = self.build(e.params["jaxpr"], ins)
      else:
        dependent = any(n.dependent for n in ins)
        outs = [Node(v.aval.shape, v.aval.dtype, eq=e, ins=ins, output=i, dependent=dependent) for i, v in enumerate(e.outvars)]
      for node in outs:
        if node.number == -1:
          node.number = self.counter
          self.counter += 1
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


class Extract:
  def __init__(self, graph, count):
    self.graph, self.count = graph, count
    self.matrices = []
    self.ops = []
    self.leaves = []
    self.cache = {}

  def op(self, fn, args, signature):
    i = len(self.ops)
    self.ops.append((fn, args, signature))
    if i > 512:
      raise Unsupported(f"{signature[0]}: local graph exceeds 512 array operations; nested extraction is needed")
    return i

  def demand(self, n, ids):
    if self.count * ids.shape[-1] > 2_000_000:
      raise Unsupported(f"{n.eq.primitive.name if n.eq else 'input'}: demand exceeds two million indices")
    ids = np.broadcast_to(ids, (self.count, ids.shape[-1]))
    ids, inv = unique(ids)
    key = (id(n), ids[0].tobytes(), ids.strides[0], None if ids.strides[0] == 0 else ids.tobytes())
    if key not in self.cache:
      self.cache[key] = self.build(n, ids)
    result = self.cache[key]
    if len(inv) == ids.shape[1] and np.array_equal(inv, np.arange(len(inv))):
      return result
    return self.op(lambda x, ix=inv: x[ix], (result,), ("gather", tuple(inv)))

  def build(self, n, ids):
    if n.value is not None:
      value = n.value.reshape(-1)[ids]
      same(value)
      v = value[0]
      return self.op(lambda v=v: jnp.asarray(v), (), ("const", v.dtype.str, v.tobytes()))
    if n.arg is not None:
      idx = len(self.leaves)
      self.leaves.append((n.arg, ids))
      return self.op(lambda: None, (), ("input", idx))
    name = n.eq.primitive.name
    p = n.eq.params
    if name == "iota":
      n.value = self.graph.constant(n)
      return self.build(n, ids)
    if name == "scatter-add":
      return self.scatter(n, ids)
    if name in ROUTING:
      return self.route(n, ids)
    if name in ELEMENTWISE:
      inputs = []
      for a in n.ins:
        mapping = np.broadcast_to(np.arange(a.size).reshape(a.shape), n.shape).reshape(-1)
        inputs.append(self.demand(a, mapping[ids]))
      fn = lambda *a: n.eq.primitive.bind(*a, **p)
      return self.op(fn, tuple(inputs), (name, repr(p)))
    if name == "custom_jvp_call":
      f = Graph.custom(n.eq)
      inputs = []
      for a in n.ins:
        mapping = np.broadcast_to(np.arange(a.size).reshape(a.shape), n.shape).reshape(-1)
        inputs.append(self.demand(a, mapping[ids]))
      return self.op(f, tuple(inputs), (name,))
    if name == "reduce_sum":
      a = n.ins[0]
      axes = p["axes"]
      free = tuple(i for i in range(len(a.shape)) if i not in axes)
      mapping = np.arange(a.size).reshape(a.shape).transpose((*free, *axes)).reshape(n.size, -1)
      ix = mapping[ids].reshape(self.count, -1)
      val = self.demand(a, ix)
      width = ids.shape[1]
      red = mapping.shape[1]
      return self.op(lambda x: x.reshape(width, red).sum(axis=1), (val,), ("reduce_sum", width, red))
    if name == "dot_general":
      return self.dot(n, ids)
    raise Unsupported(f"{name}: no backward array-demand rule")

  def route(self, n, ids):
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
      index = self.graph.constant(n.ins[1])
      with jax.ensure_compile_time_eval():
        out = np.asarray(n.eq.primitive.bind(a[0], index, **p))
    else:
      raise Unsupported(f"{name}: unsupported index routing")
    mapped = out.reshape(-1)[ids]
    if np.any(mapped < 0) or np.any(mapped >= offsets[-1]):
      raise Unsupported(f"{name}: out-of-bounds fill is unsupported")
    sources = np.searchsorted(offsets[1:], mapped, side="right")
    pattern = same(sources)
    parts = []
    order = []
    for i in np.unique(pattern):
      positions = np.flatnonzero(pattern == i)
      parts.append(self.demand(n.ins[i], mapped[:, positions] - offsets[i]))
      order.extend(positions)
    perm = np.argsort(order)
    if len(parts) == 1 and np.array_equal(perm, np.arange(len(perm))):
      return parts[0]
    return self.op(lambda *xs: jnp.concatenate(xs)[perm], tuple(parts), ("route", tuple(perm)))

  def scatter(self, n, ids):
    if n not in self.graph.scatter_maps:
      index = self.graph.constant(n.ins[1])
      p = n.eq.params

      def scatter(updates):
        return jax.lax.scatter_add(
          jnp.zeros(n.shape, np.float32),
          index,
          updates,
          p["dimension_numbers"],
          indices_are_sorted=p["indices_are_sorted"],
          unique_indices=p["unique_indices"],
          mode=p["mode"],
        )

      with jax.ensure_compile_time_eval():
        labels = np.arange(1, n.size + 1, dtype=np.float32).reshape(n.shape)
        targets = np.asarray(jax.vjp(scatter, jnp.zeros(n.ins[2].shape, np.float32))[1](labels)[0]).astype(int).reshape(-1) - 1
      buckets = [[] for _ in range(n.size)]
      for update, target in enumerate(targets):
        if target >= 0:
          buckets[target].append(update)
      counts = np.array(list(map(len, buckets)))
      table = np.zeros((n.size, counts.max(initial=0)), dtype=int)
      for row, updates in enumerate(buckets):
        table[row, : len(updates)] = updates
      self.graph.scatter_maps[n] = counts, table
    counts, table = self.graph.scatter_maps[n]
    widths = same(counts[ids])
    base = self.demand(n.ins[0], ids)
    if not widths.sum():
      return base
    indices = np.concatenate([table[ids[:, i], :width] for i, width in enumerate(widths)], axis=1)
    updates = self.demand(n.ins[2], indices)
    offsets = np.r_[0, np.cumsum(widths)]

    def add(base, updates):
      return base + jnp.stack([updates[lo:hi].sum() for lo, hi in itertools.pairwise(offsets)])

    return self.op(add, (base, updates), ("scatter-add", tuple(widths)))

  def dot(self, n, ids):
    lhs, rhs = n.ins
    p = n.eq.params
    if n.dependent and max(len(lhs.shape), len(rhs.shape)) >= 2 and any(lhs.shape[d] > 1 for d in p["dimension_numbers"][0][0]):
      self.matrices.append((n, ids))
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

    def fn(a, b):
      return jax.lax.dot_general(a.reshape(lshape), b.reshape(rshape), **p).reshape(-1)[pick]

    return self.op(fn, (li, ri), ("dot_general", lshape, rshape, repr(p), tuple(pick)))

  def finish(self):
    selected = [ids for arg, ids in self.leaves if arg == self.graph.argnums]
    if not selected:
      return None
    inputs, _ = unique(np.concatenate(selected, axis=1))
    if inputs.shape[1] > 128:
      raise Unsupported(f"local input: width {inputs.shape[1]} exceeds 128; reduction/contraction mixes the candidate axis")
    params = []
    param_leaves = []
    lookups = {}
    for i, (arg, ids) in enumerate(self.leaves):
      if arg == self.graph.argnums:
        relative = np.stack([np.searchsorted(x, y) for x, y in zip(inputs, ids)])
        lookups[i] = same(relative)
      else:
        params.append((arg, ids))
        param_leaves.append(i)

    def fn(x, *p):
      vals = []
      leaf = dict(zip(param_leaves, p))
      leaf.update((i, x[ix]) for i, ix in lookups.items())
      for op, args, sig in self.ops:
        vals.append(leaf[sig[1]] if sig[0] == "input" else op(*(vals[i] for i in args)))
      return vals[-1]

    return fn, inputs, params


def extract(f, specs, argnums, hessian):
  closed = jax.make_jaxpr(jax.grad(f, argnums=argnums) if hessian else f)(*specs)
  graph = Graph(closed, argnums)
  n = specs[argnums].size
  m = graph.output.size

  if not m:
    return [], (m, n)

  def discover(pending):
    groups = []
    while pending:
      rows = pending.pop()
      ex = Extract(graph, len(rows))
      try:
        ex.demand(graph.output, rows)
        piece = ex.finish()
      except Split as split:
        pending.extend(rows[split.labels == k] for k in np.unique(split.labels))
        if len(pending) + len(groups) > 256:
          raise Unsupported("local graph: more than 256 distinct index programs")
        continue
      if piece is not None:
        fn, inputs, params = piece
        anchor = min(ex.matrices, key=lambda a: a[0].number) if ex.matrices else None
        groups.append((fn, rows, inputs, params, anchor))
    return groups

  pieces = discover([np.arange(m)[:, None]])
  anchors, groups = {}, []
  for piece in pieces:
    _fn, rows, _inputs, _params, anchor = piece
    if anchor is None:
      groups.append(piece)
      continue
    node, indices = anchor
    for row, ids in zip(rows, indices):
      anchors.setdefault((node.number, ids.tobytes()), []).extend(row)
  by_width = {}
  for rows in anchors.values():
    by_width.setdefault(len(rows), []).append(sorted(rows))
  if by_width:
    groups.extend(discover([np.asarray(rows) for rows in by_width.values()]))
  return groups, (m, n)


def freeze(value):
  if isinstance(value, dict):
    return tuple((k, freeze(v)) for k, v in sorted(value.items()))
  if isinstance(value, (list, tuple)):
    return tuple(map(freeze, value))
  if isinstance(value, (np.ndarray, jax.Array)):
    a = np.asarray(value)
    return a.dtype.str, a.shape, a.tobytes()
  try:
    hash(value)
  except TypeError:
    return id(value)
  return value


def structured(f, specs, argnums, hessian):
  from aotax.sparse import SparseDerivative, _indices

  pieces, shape = extract(f, specs, argnums, hessian)
  groups = {}
  for fn, rows, inputs, params, _ in pieces:
    local_specs = [jax.ShapeDtypeStruct((inputs.shape[1],), specs[argnums].dtype)]
    local_specs.extend(jax.ShapeDtypeStruct((ids.shape[1],), specs[arg].dtype) for arg, ids in params)
    derivative = jax.jacfwd if inputs.shape[1] <= rows.shape[1] else jax.jacrev
    derivative_name = "jacfwd" if derivative is jax.jacfwd else "jacrev"
    closed = jax.make_jaxpr(derivative(fn))(*local_specs)
    key = (str(closed.jaxpr), freeze(closed.consts), freeze([e.params for e in closed.jaxpr.eqns]), tuple(arg for arg, _ in params), derivative_name)
    if key not in groups:
      groups[key] = (closed, [], [], [[] for _ in params], [arg for arg, _ in params], derivative_name)
    _, rr, ix, pp, _, _ = groups[key]
    rr.append(rows)
    ix.append(inputs)
    for p, (_, indices) in zip(pp, params):
      p.append(indices)
  evaluators, allrows, allcols, reports = [], [], [], []
  for closed, rr, ix, pp, args, derivative_name in groups.values():
    rows, inputs = np.concatenate(rr), np.concatenate(ix)
    params = [(arg, np.concatenate(p)) for arg, p in zip(args, pp)]
    rows, cols = np.broadcast_arrays(rows[:, :, None], inputs[:, None, :])
    keep = rows >= cols if hessian else np.ones_like(rows, dtype=bool)
    if not keep.any():
      continue
    allrows.append(rows[keep])
    allcols.append(cols[keep])
    positions = np.flatnonzero(keep)
    # Unused selected inputs in the derivative prove that it is affine.
    live = set(closed.jaxpr.outvars)
    for eq in reversed(closed.jaxpr.eqns):
      if any(v in live for v in eq.outvars):
        live.update(v for v in eq.invars if not isinstance(v, core.Literal))
    affine = closed.jaxpr.invars[0] not in live

    def make(closed, inputs, params, positions, affine):
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
        return jax.vmap(local, in_axes=tuple(axes))(x, *ps).reshape(-1)[positions]

      return values

    evaluators.append(make(closed, inputs, params, positions, affine))
    mapping = input_map(inputs)
    parameter_maps = "; ".join(f"arg {arg}: {input_map(ids)}" for arg, ids in params)
    reports.append(
      f"{len(inputs)} local maps, width {inputs.shape[1]}, input map {mapping}; "
      f"{'affine coefficients' if affine else derivative_name}; parameters {parameter_maps}"
    )
  rows = np.concatenate(allrows) if allrows else np.zeros(0, dtype=int)
  cols = np.concatenate(allcols) if allcols else np.zeros(0, dtype=int)
  order = np.lexsort((cols, rows))

  def values(*xs):
    if not evaluators:
      return jnp.zeros(0, dtype=specs[argnums].dtype)
    return jnp.concatenate([evaluate(*xs) for evaluate in evaluators])[order]

  report = "structured: backward array demand in " + ("jax.grad; local Jacobians of gradient neighborhoods" if hessian else "the array output")
  return SparseDerivative(values, _indices(rows[order]), _indices(cols[order]), shape, "\n".join([report, *reports]))


def input_map(ids):
  first = np.array2string(ids[0], threshold=16, max_line_width=120)
  if len(ids) == 1:
    return first
  stride = ids[1] - ids[0]
  if np.array_equal(ids, ids[0] + np.arange(len(ids))[:, None] * stride):
    return f"{first} + instance * {np.array2string(stride, threshold=16)}"
  return np.array2string(ids, threshold=32, max_line_width=120)
