"""Translate the pinned compiler's hardware abstraction layer (HAL) program into a synchronous C host."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from iree.compiler import ir as mlir
from iree.compiler import passmanager as mlir_passes

ir: Any = mlir
passes: Any = mlir_passes


BUFFERS = {"!hal.buffer", "!hal.buffer_view", "!util.buffer", "!hal.file"}


def binding_writes(directory: Path) -> dict[str, list[bool]]:
  result = {}

  def walk(op):
    if op.name == "hal.executable.export":
      bindings = re.findall(r"#hal.pipeline.binding<([^>]+)>", str(op.attributes["layout"]))
      result[op.attributes["sym_name"].value] = ["ReadOnly" not in b for b in bindings]
    for region in op.regions:
      for block in region.blocks:
        for child in block.operations:
          walk(child.operation)

  with ir.Context():
    for path in directory.glob("*.mlir"):
      walk(ir.Module.parse(path.read_text()).operation)
  return result


def _type(t) -> str:
  t = str(t)
  if t in BUFFERS:
    return "unsigned char*"
  if t == "!hal.executable":
    return "const executable*"
  if t == "index":
    return "size_t"
  if t.startswith("i") and t[1:].isdigit():
    return f"uint{max(8, int(t[1:]))}_t"
  if t in {"!hal.device", "!hal.allocator", "!hal.fence", "!hal.command_buffer"}:
    return "uintptr_t"
  raise ValueError(f"unsupported HAL type: {t}")


def _data(attr) -> bytes:
  if isinstance(attr, ir.DenseElementsAttr):
    return np.asarray(ir.DenseElementsAttr(attr)).tobytes()
  # Composite attributes have no Python binding; let MLIR parse their element list.
  text = str(attr)
  if text.startswith("#util.composite<") and text.endswith("]>"):
    data = b"".join(_data(a) for a in ir.ArrayAttr.parse(text[text.index("[") : -2].rstrip().removesuffix(",") + "]"))
    if len(data) != int(text.removeprefix("#util.composite<").split("xi8,", 1)[0]):
      raise ValueError("util.buffer.constant: serialized length differs from declared length")
    return data
  raise ValueError(f"util.buffer.constant: unsupported constant attribute {text[:80]}")


class _Host:
  device = "local*"
  format = "static"
  buffer_type = "unsigned char*"

  @staticmethod
  def type(t):
    return _type(t)

  def __init__(self, lib: Any, module, queries: list[str], blas, writes, kept, custom_calls):
    self.lib = lib
    self.kept = kept
    self.queries = queries
    self.blas = blas
    self.writes = writes
    self.written_inputs = set()
    self.fields = []
    self.data = []
    self.entries = {}
    self.globals = {}
    self.functions = []
    self.custom_calls = {}
    size = max((spec.scratch for spec in custom_calls.values()), default=0)
    scratch = self.storage(size) + "[0]" if size else "NULL"
    for top in module.body.operations:
      op = top.operation
      if op.name == "util.global":
        if "initial_value" in op.attributes:
          self.unsupported(op, "initialized globals")
        key = op.attributes["sym_name"].value
        self.globals[key] = f"g{len(self.globals)}"
        self.fields.append(f"{self.type(op.attributes['type'].value)} {self.globals[key]};")
      elif op.name in {"util.initializer", "util.func"}:
        if op.name == "util.func" and not op.regions[0].blocks:
          symbol = op.attributes["sym_name"].value
          if symbol not in custom_calls:
            self.unsupported(op, f"unregistered external declaration {symbol}")
          spec = custom_calls[symbol]
          t = ir.FunctionType(op.attributes["function_type"].value)
          spec.validate_hal(t.inputs, t.results)
          self.custom_calls[symbol] = (spec, scratch if spec.scratch else "NULL")
          continue
        if op.name == "util.func" and op.attributes["sym_name"].value != "main":
          self.unsupported(op, "private functions require memoization to be disabled")
        self.functions.append(op)
      elif op.name == "hal.executable":
        for child in op.regions[0].blocks[0].operations:
          if child.operation.name == "hal.executable_end":
            continue
          if child.operation.name != "hal.executable.binary" or child.attributes["format"].value != self.format:
            self.unsupported(child.operation)
      else:
        self.unsupported(op)

  @staticmethod
  def unsupported(op, reason=""):
    raise ValueError(f"aotax: unsupported {op.name}" + (f": {reason}" if reason else ""))

  def storage(self, size, count=1):
    name = f"storage{len(self.fields)}"
    self.fields.append(f"alignas(64) unsigned char {name}[{count}][{max(64, (size + 63) // 64 * 64)}];")
    return f"c->{name}"

  def add(self, buffer, offset):
    return f"({buffer} + {offset})"

  def read(self, buffer):
    return buffer

  def occupied(self, value, slot, size):
    return f"((uintptr_t){value} >= (uintptr_t){slot} && (uintptr_t){value} < (uintptr_t){slot} + {max(1, size)})"

  def constant(self, data):
    symbol = f"data{len(self.data)}"
    self.data.append(f"alignas(64) static const unsigned char {symbol}[] = {{{','.join(map(str, data)) or '0'}}};")
    return f"(unsigned char*){symbol}"

  def executable(self, target):
    query = target + "_library_query"
    if query not in self.queries:
      raise ValueError(f"aotax: no library query for {query}")
    return f"c->libraries[{self.queries.index(query)}]"

  def entry(self, path):
    entry = path[-1]
    if entry not in self.entries:
      self.entries[entry] = (len(self.entries), path[0] + "_library_query")
    return f"c->entries[{self.entries[entry][0]}]"

  def copy(self, target, source, length):
    return f"memmove({target}, {source}, {length});"

  def fill(self, target, length, pattern, width):
    return f"fill({target}, {length}, {pattern}, sizeof({width}));"

  def wait(self):
    return ""

  def begin(self):
    return ""

  def end(self):
    return ""

  def barrier(self):
    return ""

  def dispatch(self, executable, ordinal, entry, groups, pointers, lengths, constants, emit):
    x, y, z = groups
    emit("{")
    emit(f"  void* ptrs[] = {{{', '.join(pointers)}}};")
    emit(f"  const size_t lengths[] = {{{', '.join(lengths)}}};")
    emit(f"  const uint32_t constants[] = {{{', '.join(constants) or '0'}}};")
    contraction = self.blas.get(entry)
    fallback = f"if (dispatch(c, {executable}, {ordinal}, {x}, {y}, {z}, constants, ptrs, lengths)) return 1;"
    if contraction is not None:
      guard, lines = contraction.emit()
      for view in (contraction.a, contraction.b):
        if view.data:
          emit(f"  alignas(64) static const unsigned char {view.binding}[] = {{{','.join(map(str, view.data))}}};")
      emit(f"  if ({x} && {y} && {z}) {{")
      emit(f"    if ({guard}) {{")
      for line in lines:
        emit("      " + line)
      emit(f"    }} else {fallback}")
      emit("  }")
    else:
      emit("  " + fallback)
    emit("}")

  def function(self, fn, number):
    main = fn.name == "util.func"
    blocks = list(fn.regions[0].blocks)
    ops = [o.operation for b in blocks for o in b.operations]
    names, constants = {}, {}
    declarations, code = [], []
    for b in blocks:
      for v in [*b.arguments, *(v for o in b.operations for v in o.results)]:
        names[v] = f"v{len(names)}"
        declarations.append(f"  {self.type(v.type)} {names[v]} = {{0}};")

    def val(v):
      return constants.get(v, names[v])

    def signed(v):
      t = str(v.type)
      if t == "i1":
        return f"-(int8_t){val(v)}"
      cast = "intptr_t" if t == "index" else f"int{int(t[1:])}_t"
      return f"({cast}){val(v)}"

    def const(v, op=None):
      o = v.owner.operation if isinstance(v, ir.OpResult) else v.owner
      if isinstance(o, ir.Operation) and o.name == "arith.constant":
        return int(o.attributes["value"].value)
      if op is not None:
        self.unsupported(op, "requires a compile-time constant")
      return None

    def emit(s):
      if s:
        code.append("  " + s)

    def assign(o, values):
      for r, value in zip(o.results, values, strict=True):
        if str(r.type) == "i1":
          value = f"({value}) & 1"
        emit(f"{names[r]} = {value};")

    def segments(o):
      groups, start = [], 0
      for size in o.attributes["operandSegmentSizes"]:
        groups.append(list(o.operands[start : start + size]))
        start += size
      return groups

    origins = {v: {self.kept[i]} for i, v in enumerate(blocks[0].arguments)} if main else {}
    aliases = []
    for o in ops:
      for r in o.results:
        if str(r.type) in BUFFERS and o.name != "util.call":
          aliases.extend((r, v) for v in o.operands if str(v.type) in BUFFERS)
      if o.name in {"cf.br", "cf.cond_br"}:
        args = [list(o.operands)] if o.name == "cf.br" else segments(o)[1:]
        for successor, values in zip(o.successors, args, strict=True):
          aliases.extend((r, v) for r, v in zip(successor.arguments, values, strict=True) if str(r.type) in BUFFERS)
      elif o.name == "util.global.load" and str(o.results[0].type) in BUFFERS:
        aliases.extend(
          (o.results[0], store.operands[0])
          for store in ops
          if store.name == "util.global.store" and store.attributes["global"] == o.attributes["global"]
        )
    changed = True
    while changed:
      changed = False
      for r, v in aliases:
        merged = origins.get(r, set()) | origins.get(v, set())
        if merged != origins.get(r, set()):
          origins[r] = merged
          changed = True

    def written(v):
      self.written_inputs.update(origins.get(v, ()))

    # Recording and execution must stay in one block; SSA operands then keep their recorded values.
    commands = {}
    for b in blocks:
      active: dict[Any, tuple[list[Any], bool]] = {}
      for oo in b.operations:
        o = oo.operation
        if o.name == "hal.command_buffer.create":
          active[o.results[0]] = ([], False)
        elif o.name.startswith("hal.command_buffer."):
          cmd = o.operands[0]
          if cmd not in active or active[cmd][1]:
            self.unsupported(o, "command recording must follow creation in the same block")
          if o.name == "hal.command_buffer.finalize":
            active[cmd] = (active[cmd][0], True)
          else:
            active[cmd][0].append(o)
        elif o.name in {"hal.device.queue.execute", "hal.device.queue.execute.indirect"}:
          cmd = o.operands[4]
          if cmd not in active or not active[cmd][1]:
            self.unsupported(o, "execution must follow finalization in the same block")
          commands[o] = active.pop(cmd)[0]
      if active:
        self.unsupported(fn, "command buffers may not escape their recording block")
    for o in ops:
      for v in o.operands:
        if str(v.type) == "!hal.command_buffer" and o.name not in {
          "hal.command_buffer.copy_buffer",
          "hal.command_buffer.fill_buffer",
          "hal.command_buffer.dispatch",
          "hal.command_buffer.execution_barrier",
          "hal.command_buffer.finalize",
          "hal.device.queue.execute",
          "hal.device.queue.execute.indirect",
        }:
          self.unsupported(o, "command buffer escapes")
    # Extend recorded buffer uses through execution, then compute SSA liveness over the CFG.
    uses = {o: set(o.operands) for o in ops}
    for o, recorded in commands.items():
      uses[o].update(v for command in recorded for v in command.operands)
    live_in = {b: set() for b in blocks}
    live_after = {}
    changed = True
    while changed:
      changed = False
      for b in reversed(blocks):
        live = set().union(*(live_in[s] for s in b.operations[-1].operation.successors))
        for oo in reversed(list(b.operations)):
          o = oo.operation
          live_after[o] = live.copy()
          live.difference_update(o.results)
          live.update(uses[o])
        live.difference_update(b.arguments)
        if live != live_in[b]:
          live_in[b] = live
          changed = True

    def cyclic(b):
      todo = list(b.operations[-1].operation.successors)
      seen = set()
      while todo:
        s = todo.pop()
        if s == b:
          return True
        if s not in seen:
          seen.add(s)
          todo.extend(s.operations[-1].operation.successors)
      return False

    def allocate(o, result, size):
      live = (
        sorted((v for v in live_after[o] | uses[o] if str(v.type) in BUFFERS and v not in o.results), key=lambda v: names[v])
        if cyclic(o.block)
        else []
      )
      pool = self.storage(size, len(live) + 1)
      if live:
        emit("{")
        emit("  size_t slot = 0;")
        checks = [self.occupied(names[v], f"{pool}[slot]", size) for v in live]
        emit(f"  while ({' || '.join(checks)}) ++slot;")
        emit(f"{names[result]} = {pool}[slot];")
        emit("}")
      else:
        emit(f"{names[result]} = {pool}[0];")

    direct = {}
    returned = {}
    if main and self.format == "static":
      returns = [o for o in ops if o.name == "util.return"]
      if len(returns) == 1:
        for i, v in enumerate(returns[0].operands):
          view = v.owner.operation if isinstance(v, ir.OpResult) else v.owner
          if isinstance(view, ir.Operation) and view.name == "hal.buffer_view.create":
            buf = view.operands[0]
            alloc = buf.owner.operation if isinstance(buf, ir.OpResult) else buf.owner
            if (
              isinstance(alloc, ir.Operation)
              and alloc.name == "hal.device.queue.alloca"
              and not cyclic(alloc.block)
              and const(view.operands[1]) == 0
              and const(alloc.operands[-1]) == self.lib.out_avals[i].size * self.lib.out_avals[i].dtype.itemsize
            ):
              returned[buf] = returned.get(buf, 0) + 1
              direct[buf] = i if returned[buf] == 1 else None

    def binding(v, offset, table, write=False):
      if str(v.type) == "index":
        if table is None:
          self.unsupported(fn, "indirect binding without a binding table")
        index = const(v, v.owner.operation if isinstance(v, ir.OpResult) else fn)
        buf, base, _ = table[index]
        if write:
          written(buf)
        return self.add(val(buf), f"{val(base)} + {val(offset)}")
      if write:
        written(v)
      return self.add(val(v), val(offset))

    def dispatch(o, table):
      g = segments(o)
      executable, ordinal = val(g[1][0]), val(g[2][0])
      x, y, z = [val(g[i][0]) for i in (3, 4, 5)]
      lookup = g[2][0].owner.operation
      if lookup.name != "hal.executable.lookup.function":
        self.unsupported(o, "dispatch requires a known entry point")
      entry = lookup.attributes["entry_point"].value[-1]
      if entry not in self.writes:
        self.unsupported(o, f"missing binding layout for {entry}")
      pointers = [binding(v, off, table, write) for v, off, write in zip(g[7], g[8], self.writes[entry], strict=True)]
      self.dispatch(executable, ordinal, entry, (x, y, z), pointers, [val(v) for v in g[9]], [val(v) for v in g[6]], emit)

    def command(o, table):
      a = list(o.operands)
      if o.name == "hal.command_buffer.dispatch":
        dispatch(o, table)
      elif o.name == "hal.command_buffer.copy_buffer":
        emit(self.copy(binding(a[3], a[4], table, True), binding(a[1], a[2], table), val(a[5])))
      elif o.name == "hal.command_buffer.fill_buffer":
        emit(self.fill(binding(a[1], a[2], table, True), val(a[3]), val(a[4]), self.type(a[4].type)))
      elif o.name == "hal.command_buffer.execution_barrier":
        emit(self.barrier())
      else:
        self.unsupported(o)

    def branch(b, args):
      emit("{")
      for i, v in enumerate(args):
        emit(f"  {self.type(v.type)} t{i} = {val(v)};")
      for i, v in enumerate(b.arguments):
        emit(f"  {names[v]} = t{i};")
      emit(f"  goto b{blocks.index(b)};")
      emit("}")

    if main:
      for i, arg in enumerate(blocks[0].arguments):
        emit(f"{names[arg]} = inputs[{self.kept[i]}];")
    for bi, b in enumerate(blocks):
      emit(f"b{bi}:;")
      for oo in b.operations:
        o = oo.operation
        a = [val(v) for v in o.operands]
        n = o.name
        if n == "arith.constant":
          constants[o.results[0]] = f"(({self.type(o.results[0].type)}){int(o.attributes['value'].value)})"
        elif n in {
          "arith.addi",
          "arith.subi",
          "arith.muli",
          "arith.andi",
          "arith.ori",
          "arith.xori",
          "arith.divui",
          "arith.remui",
          "arith.shli",
          "arith.shrui",
        }:
          operator = {
            "addi": "+",
            "subi": "-",
            "muli": "*",
            "andi": "&",
            "ori": "|",
            "xori": "^",
            "divui": "/",
            "remui": "%",
            "shli": "<<",
            "shrui": ">>",
          }[n.split(".")[1]]
          left = f"(uint64_t){a[0]}" if n == "arith.muli" else a[0]
          assign(o, [f"{left} {operator} {a[1]}"])
        elif n in {"arith.minui", "arith.maxui", "arith.minsi", "arith.maxsi"}:
          left, right = [signed(v) for v in o.operands] if n.endswith("si") else a
          assign(o, [f"{left} {'<' if '.min' in n else '>'} {right} ? {a[0]} : {a[1]}"])
        elif n in {"arith.index_cast", "arith.extsi"}:
          assign(o, [f"({self.type(o.results[0].type)})({signed(o.operands[0])})"])
        elif n in {"arith.index_castui", "arith.extui", "arith.trunci"}:
          assign(o, [f"({self.type(o.results[0].type)}){a[0]}"])
        elif n in {"arith.cmpi", "util.cmp.eq"}:
          pred = int(o.attributes["predicate"].value) if n == "arith.cmpi" else 0
          operator = ["==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">="][pred]
          left, right = [signed(v) for v in o.operands] if 2 <= pred <= 5 else a
          assign(o, [f"{left} {operator} {right}"])
        elif n == "arith.select":
          assign(o, [f"{a[0]} ? {a[1]} : {a[2]}"])
        elif n == "cf.br":
          branch(o.successors[0], o.operands)
        elif n == "cf.cond_br":
          g = segments(o)
          emit(f"if ({a[0]}) {{")
          branch(o.successors[0], g[1])
          emit("} else {")
          branch(o.successors[1], g[2])
          emit("}")
        elif n == "util.null":
          assign(o, [f"({self.type(o.results[0].type)}){{0}}"])
        elif n == "util.global.load":
          assign(o, [f"c->{self.globals[o.attributes['global'].value]}"])
        elif n == "util.global.store":
          emit(f"c->{self.globals[o.attributes['global'].value]} = {a[0]};")
        elif n == "util.buffer.constant":
          data = _data(o.attributes["value"])
          assign(o, [self.constant(data)])
        elif n == "hal.devices.count" or n in {
          "hal.devices.get",
          "hal.device.allocator",
          "hal.fence.create",
          "hal.fence.join",
          "hal.buffer_usage",
          "hal.memory_type",
          "hal.element_type",
          "hal.encoding_type",
        }:
          assign(o, ["1"])
        elif n == "hal.device.query":
          category, key = o.attributes["category"].value, o.attributes["key"].value
          if (category, key) not in {("hal.device.id", self.device), ("hal.executable.format", self.format)}:
            self.unsupported(o, f"device query {category}/{key}")
          assign(o, ["1", "1"])
        elif n == "hal.executable.create":
          assign(o, [self.executable(o.attributes["executable_target"].value[0])])
        elif n == "hal.executable.lookup.function":
          constants[o.results[0]] = self.entry(o.attributes["entry_point"].value)
        elif n in {"hal.device.queue.alloca", "hal.allocator.allocate"}:
          size = const(o.operands[-1], o)
          output = direct.get(o.results[0])
          if output is not None:
            assign(o, [f"outputs[{output}]"])
            continue
          allocate(o, o.results[0], size)
        elif n == "hal.allocator.import":
          assign(o, ["1", self.add(a[4], a[5])])
        elif n == "hal.ex.file.from_memory":
          assign(o, [self.add(a[2], a[3])])
        elif n in {"hal.device.queue.read", "hal.device.queue.copy"}:
          written(o.operands[6])
          emit(self.begin())
          emit(self.copy(self.add(a[6], a[7]), self.add(a[4], a[5]), a[8]))
          emit(self.end())
        elif n == "hal.device.queue.fill":
          written(o.operands[4])
          emit(self.begin())
          emit(self.fill(self.add(a[4], a[5]), a[6], a[7], self.type(o.operands[7].type)))
          emit(self.end())
        elif n in {"hal.buffer_view.buffer", "util.optimization_barrier"}:
          assign(o, a)
        elif n in {"hal.buffer_view.create", "hal.buffer.subspan"}:
          assign(o, [self.add(a[0], a[1])])
        elif n == "hal.buffer.load":
          emit(self.wait())
          source = self.add(a[0], a[1])
          emit(f"memcpy(&{names[o.results[0]]}, {self.read(source)}, sizeof({names[o.results[0]]}));")
        elif n in {"hal.device.queue.execute", "hal.device.queue.execute.indirect"}:
          count = (len(o.operands) - 5) // 3
          table = (
            list(zip(o.operands[5 : 5 + count], o.operands[5 + count : 5 + 2 * count], o.operands[5 + 2 * count :], strict=True))
            if n.endswith(".indirect")
            else None
          )
          emit(self.begin())
          for recorded in commands[o]:
            command(recorded, table)
          emit(self.end())
        elif n.startswith("hal.command_buffer.") or n in {"hal.buffer.assert", "hal.buffer_view.assert", "hal.device.queue.dealloca"}:
          pass
        elif n == "hal.fence.await":
          emit(self.wait())
          assign(o, ["0"])
        elif n == "util.status.check_ok":
          emit(f"if ({a[0]}) return fail({json.dumps(o.attributes['message'].value)});")
        elif n == "util.call":
          symbol = o.attributes["callee"].value
          if symbol not in self.custom_calls:
            self.unsupported(o, f"unregistered call {symbol}")
          spec, scratch = self.custom_calls[symbol]
          spec.validate_hal([v.type for v in o.operands], [v.type for v in o.results])
          for result, tensor in zip(o.results, spec.results, strict=True):
            allocate(o, result, tensor.size)
          if self.device == "metal":
            emit(self.wait())
          pointers = [*(self.read(p) for p in (*a, *(names[r] for r in o.results))), scratch]
          emit(f"if ({symbol}({', '.join(pointers)})) return fail({json.dumps(f'custom call {spec.target} failed')});")
        elif n == "util.return":
          if main:
            for i, value in enumerate(a):
              emit(f"results[{i}] = {value};")
          emit("return 0;")
        else:
          self.unsupported(o)
    parameters = f", {self.buffer_type}* inputs, {self.buffer_type}* outputs, {self.buffer_type}* results" if main else ""
    return f"static int run{number}({self.lib.name}_context* c{parameters}) {{\n" + "\n".join(declarations + code) + "\n}\n"


def host(lib: Any, text: str, queries: list[str], signature: str, blas, writes, kept, custom_calls, device="cpu") -> str:
  from aotax.metal import MetalHost

  with ir.Context():
    module = ir.Module.parse(text)
    # Embedded constants import directly; their allocation fallback is unreachable.
    imports = [
      o.operation
      for top in module.body.operations
      if top.operation.name in {"util.initializer", "util.func"}
      for block in top.regions[0].blocks
      for o in list(block.operations)
      if o.operation.name == "hal.allocator.import"
    ]
    for op in imports:
      imported = ir.Operation.create(
        "arith.constant",
        results=[ir.IntegerType.get_signless(1)],
        attributes={"value": ir.BoolAttr.get(True)},
        ip=ir.InsertionPoint(op),
        loc=op.location,
      )
      op.results[0].replace_all_uses_with(imported.results[0])
    if imports:
      passes.PassManager.parse("builtin.module(canonicalize)").run(module.operation)
    if device == "metal":
      h = MetalHost(lib, module, queries, blas, writes, kept, custom_calls)
    else:
      h = _Host(lib, module, queries, blas, writes, kept, custom_calls)
    functions = [h.function(fn, i) for i, fn in enumerate(h.functions)]
    if isinstance(h, MetalHost):
      return h.render(functions, signature)
  n = lib.name
  ins, outs = len(lib.in_avals), len(lib.out_avals)
  stage_inputs = [h.storage(a.size * a.dtype.itemsize) for a in lib.in_avals]
  stage_outputs = [h.storage(a.size * a.dtype.itemsize) for a in lib.out_avals]
  sizes = [a.size * a.dtype.itemsize for a in (*lib.in_avals, *lib.out_avals)]
  setup = []
  for i, query in enumerate(queries):
    setup.append(f"  c->libraries[{i}] = (const executable*){query}(6, &env);")
    setup.append(
      f"  if (!c->libraries[{i}] || c->libraries[{i}]->header->version != 6 || c->libraries[{i}]->imports.count || c->libraries[{i}]->constants.count) goto incompatible;"
    )
    setup.append(
      f"  for (uint32_t j = 0; j < c->libraries[{i}]->exports.count; ++j) {{\n    const dispatch_attrs* a = &c->libraries[{i}]->exports.attrs[j];\n    if (a->flags & ~UINT64_C(1)) goto incompatible;\n    if ((size_t)a->local_memory_pages * 4096 > c->local_size) c->local_size = (size_t)a->local_memory_pages * 4096;\n  }}"
    )
  for entry, (i, query) in h.entries.items():
    setup.append(
      f"  c->entries[{i}] = find_entry(c->libraries[{queries.index(query)}], {json.dumps(entry)});\n  if (c->entries[{i}] < 0) goto incompatible;"
    )
  initializers = []
  for i, fn in enumerate(h.functions):
    if fn.name == "util.initializer":
      initializers.append(f"  if (!status) status = run{i}(c);")
  main = next(i for i, fn in enumerate(h.functions) if fn.name == "util.func")
  stages = []
  for i, storage in enumerate(stage_inputs):
    overlap = " || ".join(f"overlap(in{i}, {sizes[i]}, out{j}, {sizes[ins + j]})" for j in range(outs)) or "0"
    copy = f"memcpy({storage}[0], in{i}, {sizes[i]}); inputs[{i}] = {storage}[0];"
    stages.append(f"  {copy}" if i in h.written_inputs else f"  if (((uintptr_t)in{i} & 63) || {overlap}) {{ {copy} }}")
  for i, storage in enumerate(stage_outputs):
    overlap = " || ".join(f"overlap(out{i}, {sizes[ins + i]}, out{j}, {sizes[ins + j]})" for j in range(outs) if j != i) or "0"
    stages.append(f"  if (((uintptr_t)out{i} & 63) || {overlap}) outputs[{i}] = {storage}[0];")
  copies = [f"  if (results[{i}] != (unsigned char*)out{i}) memmove(out{i}, results[{i}], {sizes[ins + i]});" for i in range(outs)]
  pointers = [*(f"in{i}" for i in range(ins)), *(f"out{i}" for i in range(outs))]
  nulls = " || ".join(["!context", *(f"!{p}" for p, size in zip(pointers, sizes, strict=True) if size)])
  abi = Path(__file__).with_name("abi.h").read_text()
  return f'''{abi}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
{"#define ACCELERATE_NEW_LAPACK\n#include <Accelerate/Accelerate.h>" if blas or any(s.target.startswith("lapack_") and s.foreign is None for s in custom_calls.values()) else ""}
#include "{n}.h"

{chr(10).join(f"extern const library_header** {q}(uint32_t, const environment*);" for q in queries)}
alignas(16) static const environment env = {{0}};
{chr(10).join(h.data)}
struct {n}_context {{
  const executable* libraries[{max(1, len(queries))}];
  int entries[{max(1, len(h.entries))}];
  void* local_memory;
  size_t local_size;
  {chr(10).join(h.fields)}
}};

{fpu(lib.approximate_math)}

static int fail(const char* message) {{ fprintf(stderr, "{n}: %s\\n", message); return 1; }}
static int find_entry(const executable* e, const char* name) {{
  for (uint32_t i = 0; i < e->exports.count; ++i) if (!strcmp(e->exports.names[i], name)) return i;
  return -1;
}}
static int overlap(const void* a, size_t an, const void* b, size_t bn) {{
  return (uintptr_t)a <= (uintptr_t)b ? (uintptr_t)b - (uintptr_t)a < an : (uintptr_t)a - (uintptr_t)b < bn;
}}
static void fill(unsigned char* target, size_t length, uint64_t pattern, size_t width) {{
  if (width == 1) {{ memset(target, pattern, length); return; }}
  for (size_t i = 0; i < length; i += width) memcpy(target + i, &pattern, width);
}}
static int dispatch({n}_context* c, const executable* e, size_t ordinal, uint32_t x, uint32_t y, uint16_t z,
                    const uint32_t* constants, void* const* bindings, const size_t* lengths) {{
  const dispatch_attrs* a = &e->exports.attrs[ordinal];
  alignas(16) dispatch_state state = {{a->size_x, a->size_y, a->size_z, a->constant_count, x, y, z, 1, a->binding_count, constants, bindings, lengths}};
  alignas(16) workgroup_state wg = {{.local_memory = c->local_memory, .local_memory_size = c->local_size}};
  for (wg.z = 0; wg.z < z; ++wg.z)
    for (wg.y = 0; wg.y < y; ++wg.y)
      for (wg.x = 0; wg.x < x; ++wg.x)
        if (e->exports.functions[ordinal](&env, &state, &wg)) return fail("dispatch failed");
  return 0;
}}
{chr(10).join(spec.adapter(symbol) for symbol, (spec, _) in h.custom_calls.items())}
{chr(10).join(functions)}
int {n}_create({n}_context** context) {{
  if (!context) return fail("null context pointer");
  *context = NULL;
  {n}_context* c = aligned_alloc(64, (sizeof(*c) + 63) / 64 * 64);
  if (!c) return fail("allocating context");
  memset(c, 0, sizeof(*c));
{chr(10).join(setup)}
  if (c->local_size) {{
    c->local_memory = aligned_alloc(64, c->local_size);
    if (!c->local_memory) {{ {n}_destroy(c); return fail("allocating workgroup memory"); }}
  }}
  uint64_t fp = fpu_enter();
  int status = 0;
{chr(10).join(initializers)}
  fpu_leave(fp);
  if (status) {{ {n}_destroy(c); return 1; }}
  *context = c;
  return 0;
incompatible:
  {n}_destroy(c);
  return fail("unsupported executable library ABI, imports, constants or dispatch flags");
}}

{signature} {{
  if ({nulls}) return fail("null argument");
  {n}_context* c = context;
  unsigned char* inputs[{max(1, ins)}] = {{{", ".join(f"(unsigned char*)in{i}" for i in range(ins)) or "0"}}};
  unsigned char* outputs[{max(1, outs)}] = {{{", ".join(f"(unsigned char*)out{i}" for i in range(outs)) or "0"}}};
  unsigned char* results[{max(1, outs)}] = {{0}};
{chr(10).join(stages)}
  uint64_t fp = fpu_enter();
  int status = run{main}(c, inputs, outputs, results);
  fpu_leave(fp);
  if (status) return 1;
{chr(10).join(copies)}
  return 0;
}}
void {n}_destroy({n}_context* c) {{ if (c) {{ free(c->local_memory); free(c); }} }}
'''


def fpu(approximate):
  return f"""#if defined(__aarch64__)
#define FPU_MASK UINT64_C(0x1080000)
static uint64_t fpu_get(void) {{ uint64_t state; __asm__ volatile("mrs %0, fpcr" : "=r"(state)); return state; }}
static void fpu_set(uint64_t state) {{ __asm__ volatile("msr fpcr, %0" : : "r"(state)); }}
#elif defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define FPU_MASK UINT64_C(0x8040)
static uint64_t fpu_get(void) {{ return _mm_getcsr(); }}
static void fpu_set(uint64_t state) {{ _mm_setcsr((uint32_t)state); }}
#else
#define FPU_MASK UINT64_C(0)
static uint64_t fpu_get(void) {{ return 0; }}
static void fpu_set(uint64_t state) {{ (void)state; }}
#endif
#define FPU_MODE(state) ({"(state) | FPU_MASK" if approximate else "(state) & ~FPU_MASK"})
static uint64_t fpu_enter(void) {{
  uint64_t state = fpu_get();
  if (state != FPU_MODE(state)) fpu_set(FPU_MODE(state));
  return state;
}}
static void fpu_leave(uint64_t state) {{ if (fpu_get() != state) fpu_set(state); }}"""
