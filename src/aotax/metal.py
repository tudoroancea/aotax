"""Metal executable schema and device hooks for the shared HAL translator."""

import json
import struct

from aotax.host import BUFFERS, _data, _Host, _type, ir


def validate(text):
  with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text)

    def walk(op):
      if op.name == "stablehlo.custom_call":
        raise ValueError(f"aotax: Metal does not support custom calls: {op.attributes['call_target_name'].value}")
      for value in (*op.operands, *op.results):
        t = value.type
        if isinstance(t, ir.RankedTensorType) and str(ir.RankedTensorType(t).element_type) == "f64":
          raise TypeError(f"aotax: Metal does not support float64 intermediates ({op.name}: {t})")
      for region in op.regions:
        for block in region.blocks:
          for child in block.operations:
            walk(child.operation)

    walk(module.operation)


class _FlatBuffer:
  def __init__(self, data, position):
    self.data, self.position = data, position

  def field(self, index):
    vtable = self.position - struct.unpack_from("<i", self.data, self.position)[0]
    size = struct.unpack_from("<H", self.data, vtable)[0]
    offset = struct.unpack_from("<H", self.data, vtable + 4 + 2 * index)[0] if 4 + 2 * index < size else 0
    return self.position + offset if offset else 0

  def scalar(self, index, fmt="<I"):
    p = self.field(index)
    return struct.unpack_from(fmt, self.data, p)[0] if p else 0

  def ref(self, index):
    p = self.field(index)
    return p + struct.unpack_from("<I", self.data, p)[0] if p else 0

  def string(self, index):
    p = self.ref(index)
    return self.data[p + 4 : p + 4 + struct.unpack_from("<I", self.data, p)[0]].decode() if p else ""

  def tables(self, index):
    p = self.ref(index)
    size = struct.unpack_from("<I", self.data, p)[0] if p else 0
    return [_FlatBuffer(self.data, p + 4 + 4 * k + struct.unpack_from("<I", self.data, p + 4 + 4 * k)[0]) for k in range(size)]

  def scalars(self, index):
    p = self.ref(index)
    size = struct.unpack_from("<I", self.data, p)[0] if p else 0
    return [struct.unpack_from("<Q", self.data, p + 4 + 8 * k)[0] for k in range(size)]


def _pipelines(binary):
  # IREE 3.12.0: runtime/src/iree/schemas/metal_executable_def.fbs, after its 64-byte header.
  data = binary[64:]
  if data[4:8] != b"MTL1":
    raise ValueError("aotax: unsupported Metal executable schema (expected IREE 3.12 MTL1)")
  root = _FlatBuffer(data, struct.unpack_from("<I", data)[0])
  libraries = [_FlatBuffer(data, lib.ref(0)) for lib in root.tables(1)]
  result = []
  for p in root.tables(0):
    source = libraries[p.scalar(0)]
    result.append(
      {
        "name": p.string(8) or p.string(1),
        "entry": p.string(1),
        "source": source.string(1),
        "version": source.scalar(0),
        "threads": struct.unpack_from("<III", data, p.field(3)),
        "constants": p.scalar(5),
        "bindings": p.scalars(6),
      }
    )
  return result


class MetalHost(_Host):
  device = "metal"
  format = "metal-msl-fb"
  buffer_type = "view"

  def __init__(self, *args):
    self.buffers = []
    self.pipelines = []
    self.sites = []
    self.has_copies = False
    self.setup = []
    super().__init__(*args)
    module = args[1]
    self.targets = {}
    for top in module.body.operations:
      op = top.operation
      if op.name != "hal.executable":
        continue
      target = op.attributes["sym_name"].value
      self.targets[target] = len(self.pipelines) + 1
      for child in op.regions[0].blocks[0].operations:
        if child.operation.name != "hal.executable.binary":
          continue
        for pipeline in _pipelines(_data(child.attributes["data"])):
          self.entries[(target, pipeline["name"])] = len(self.pipelines)
          self.pipelines.append(pipeline)

  @staticmethod
  def type(t):
    if str(t) in BUFFERS:
      return "view"
    return "size_t" if str(t) == "!hal.executable" else _type(t)

  def storage(self, size, count=1):
    name = f"storage{len(self.fields)}"
    self.fields.append(f"view {name}[{count}];")
    for i in range(count):
      self.buffers.append((max(4, size), None))
      self.setup.append(f"c->{name}[{i}] = (view){{{len(self.buffers)}, 0}};")
    return f"c->{name}"

  def constant(self, data):
    self.buffers.append((max(4, len(data)), data))
    return f"((view){{{len(self.buffers)}, 0}})"

  def add(self, buffer, offset):
    return f"offset({buffer}, {offset})"

  def read(self, buffer):
    return f"contents(c, {buffer})"

  def occupied(self, value, slot, size):
    return f"{value}.buffer == {slot}.buffer"

  def executable(self, target):
    return str(self.targets[target])

  def entry(self, path):
    return str(self.entries[(path[0], path[-1])])

  def copy(self, target, source, length):
    self.has_copies = True
    return f"copy(c, {target}, {source}, {length});"

  def fill(self, target, length, pattern, width):
    seed = self.storage(8) + "[0]"
    return f"fill(c, {target}, {length}, {pattern}, sizeof({width}), {seed});"

  def wait(self):
    return "if (await_device(c)) return 1;"

  def begin(self):
    return "if (begin(c)) return 1;"

  def end(self):
    return "submit(c);"

  def barrier(self):
    return "barrier(c);"

  def dispatch(self, executable, ordinal, entry, groups, pointers, lengths, constants, emit):
    count = len(self.sites)
    self.sites.append((ordinal, len(pointers), len(constants)))
    emit("{")
    emit(f"  const view bindings[] = {{{', '.join(pointers) or '{0}'}}};")
    emit(f"  const uint32_t constants[] = {{{', '.join(constants) or '0'}}};")
    emit(f"  dispatch(c, &c->sites[{count}], {ordinal}, MTLSizeMake({', '.join(groups)}), bindings, constants);")
    emit("}")

  def render(self, functions, signature):
    n = self.lib.name
    ins, outs = len(self.lib.in_avals), len(self.lib.out_avals)
    stages = [self.storage(a.size * a.dtype.itemsize) + "[0]" for a in self.lib.in_avals]
    if self.has_copies:
      scratch = self.storage(max(size for size, _ in self.buffers)) + "[0]"
      self.setup.append(f"c->scratch = {scratch};")
    sizes = [a.size * a.dtype.itemsize for a in (*self.lib.in_avals, *self.lib.out_avals)]
    pointers = [*(f"in{i}" for i in range(ins)), *(f"out{i}" for i in range(outs))]
    nulls = " || ".join(["!context", *(f"!{p}" for p, size in zip(pointers, sizes, strict=True) if size)])
    setup = list(self.setup)
    for i, (size, data) in enumerate(self.buffers, 1):
      setup.append(f"c->buffers[{i}] = keep(c, [c->device newBufferWithLength:{size} options:MTLResourceStorageModeShared]);")
      setup.append(f'if (!c->buffers[{i}]) return fail("allocating Metal buffer");')
      setup.append(f"[c->residency addAllocation:c->buffers[{i}]];")
      if data:
        symbol = f"data{i}"
        self.data.append(f"static const unsigned char {symbol}[] = {{{','.join(map(str, data))}}};")
        setup.append(f"memcpy(c->buffers[{i}].contents, {symbol}, {len(data)});")
    for i, p in enumerate(self.pipelines):
      setup.extend(
        [
          "{",
          "MTLCompileOptions* options = [MTLCompileOptions new];",
          f"options.languageVersion = (MTLLanguageVersion){p['version']};",
          "options.mathMode = MTLMathModeSafe;",
          f"id<MTLLibrary> lib = [c->device newLibraryWithSource:@{json.dumps(p['source'])} options:options error:&error];",
          'if (!lib) return fail(error.localizedDescription.UTF8String ?: "compiling Metal source");',
          f"id<MTLFunction> fn = [lib newFunctionWithName:@{json.dumps(p['entry'])}];",
          'if (!fn) return fail("missing Metal entry point");',
          f"c->pipelines[{i}] = keep(c, [c->device newComputePipelineStateWithFunction:fn error:&error]);",
          f'if (!c->pipelines[{i}]) return fail(error.localizedDescription.UTF8String ?: "compiling Metal pipeline");',
          f"c->threads[{i}] = MTLSizeMake({', '.join(map(str, p['threads']))});",
          f"c->encoders[{i}] = keep(c, [fn newArgumentEncoderWithBufferIndex:0]);",
          "}",
        ]
      )
    for i, (ordinal, bindings, constants) in enumerate(self.sites):
      p = self.pipelines[int(ordinal)]
      if bindings != len(p["bindings"]) or constants != p["constants"]:
        raise ValueError("aotax: Metal dispatch layout differs from executable schema")
      setup.extend(
        [
          "{",
          f"site* s = &c->sites[{i}];",
          f"s->count = {bindings}; s->constant_count = {constants};",
          f"s->encoder = c->encoders[{ordinal}];",
          "s->arguments = keep(c, [c->device newBufferWithLength:MAX(16, s->encoder.encodedLength) options:MTLResourceStorageModeShared]);",
          "s->constants = keep(c, [c->device newBufferWithLength:MAX(4, s->constant_count * 4) options:MTLResourceStorageModeShared]);",
          "MTL4ArgumentTableDescriptor* desc = [MTL4ArgumentTableDescriptor new];",
          "desc.maxBufferBindCount = 4; desc.initializeBindings = YES;",
          "s->table = keep(c, [c->device newArgumentTableWithDescriptor:desc error:&error]);",
          'if (!s->arguments || !s->constants || !s->table) return fail("allocating Metal arguments");',
          "[s->table setAddress:s->arguments.gpuAddress atIndex:0];",
          "[s->table setAddress:s->constants.gpuAddress atIndex:3];",
          "[c->residency addAllocation:s->arguments]; [c->residency addAllocation:s->constants];",
          "}",
        ]
      )
    setup.extend(["[c->residency commit];", "[c->queue addResidencySet:c->residency];"])
    for i, fn in enumerate(self.functions):
      if fn.name == "util.initializer":
        setup.append(f"if (run{i}(c)) return 1;")
    setup.append("return await_device(c);")
    main = next(i for i, fn in enumerate(self.functions) if fn.name == "util.func")
    max_bindings = max((len(p["bindings"]) for p in self.pipelines), default=1)
    return f'''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "{n}.h"

typedef struct {{ size_t buffer, offset; }} view;
typedef struct {{
  __unsafe_unretained id<MTLArgumentEncoder> encoder;
  __unsafe_unretained id<MTLBuffer> arguments, constants;
  __unsafe_unretained id<MTL4ArgumentTable> table;
  size_t count, constant_count;
  int bound;
  view bindings[{max(1, max_bindings)}];
}} site;
struct {n}_context {{
  void* owner;
  __unsafe_unretained NSMutableArray* owned;
  __unsafe_unretained id<MTLDevice> device;
  __unsafe_unretained id<MTL4CommandQueue> queue;
  __unsafe_unretained id<MTL4CommandAllocator> allocator;
  __unsafe_unretained id<MTL4CommandBuffer> command;
  __strong id<MTL4ComputeCommandEncoder> encoder;
  __strong NSError* error;
  __unsafe_unretained dispatch_semaphore_t feedback;
  __unsafe_unretained id<MTLResidencySet> residency;
  __unsafe_unretained id<MTLSharedEvent> event;
  __unsafe_unretained id<MTLBuffer> buffers[{len(self.buffers) + 1}];
  __unsafe_unretained id<MTLComputePipelineState> pipelines[{max(1, len(self.pipelines))}];
  __unsafe_unretained id<MTLArgumentEncoder> encoders[{max(1, len(self.pipelines))}];
  MTLSize threads[{max(1, len(self.pipelines))}];
  site sites[{max(1, len(self.sites))}];
  uint64_t submitted, completed;
  view scratch;
  {chr(10).join(self.fields)}
}};
{chr(10).join(self.data)}
static int fail(const char* message) {{ fprintf(stderr, "{n}: %s\\n", message); return 1; }}
static id keep({n}_context* c, id object) {{ if (object) [c->owned addObject:object]; return object; }}
static view offset(view v, size_t offset) {{ v.offset += offset; return v; }}
static void* contents({n}_context* c, view v) {{ return (unsigned char*)c->buffers[v.buffer].contents + v.offset; }}
static int await_device({n}_context* c) {{
  if (c->completed == c->submitted) return c->error ? fail(c->error.localizedDescription.UTF8String) : 0;
  if (![c->event waitUntilSignaledValue:c->submitted timeoutMS:UINT64_MAX]) return fail("waiting for Metal execution");
  dispatch_semaphore_wait(c->feedback, DISPATCH_TIME_FOREVER);
  c->completed = c->submitted;
  return c->error ? fail(c->error.localizedDescription.UTF8String) : 0;
}}
static int begin({n}_context* c) {{
  if (await_device(c)) return 1;
  [c->allocator reset];
  [c->command beginCommandBufferWithAllocator:c->allocator];
  c->encoder = [c->command computeCommandEncoder];
  return 0;
}}
static void submit({n}_context* c) {{
  [c->encoder endEncoding]; c->encoder = nil;
  [c->command endCommandBuffer];
  id<MTL4CommandBuffer> command = c->command;
  ++c->submitted;
  MTL4CommitOptions* options = [MTL4CommitOptions new];
  [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {{
    c->error = feedback.error;
    if (c->error) c->event.signaledValue = c->submitted;
    dispatch_semaphore_signal(c->feedback);
  }}];
  [c->queue commit:&command count:1 options:options];
  [c->queue signalEvent:c->event value:c->submitted];
}}
static void barrier({n}_context* c) {{
  [c->encoder barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit
    beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
}}
static void copy({n}_context* c, view target, view source, size_t length) {{
  if (!length || (target.buffer == source.buffer && target.offset == source.offset)) return;
  if (target.buffer == source.buffer &&
      (target.offset < source.offset ? source.offset - target.offset : target.offset - source.offset) < length) {{
    copy(c, c->scratch, source, length); source = c->scratch;
  }}
  barrier(c);
  [c->encoder copyFromBuffer:c->buffers[source.buffer] sourceOffset:source.offset
    toBuffer:c->buffers[target.buffer] destinationOffset:target.offset size:length];
  barrier(c);
}}
static void fill({n}_context* c, view target, size_t length, uint64_t pattern, size_t width, view seed) {{
  int repeated = 1;
  for (size_t i = 1; i < width; ++i)
    if (((pattern >> (8 * i)) & 255) != (pattern & 255)) repeated = 0;
  if (!repeated) {{
    memcpy(contents(c, seed), &pattern, width);
    for (size_t n = 0; n < length;) {{
      size_t count = MIN(n ? n : width, length - n);
      copy(c, offset(target, n), n ? target : seed, count); n += count;
    }}
    return;
  }}
  if (length) {{
    barrier(c);
    [c->encoder fillBuffer:c->buffers[target.buffer] range:NSMakeRange(target.offset, length) value:pattern & 255];
    barrier(c);
  }}
}}
static void dispatch({n}_context* c, site* s, size_t ordinal, MTLSize groups, const view* bindings, const uint32_t* constants) {{
  if (!groups.width || !groups.height || !groups.depth) return;
  if (!s->bound || memcmp(s->bindings, bindings, sizeof(view) * s->count)) {{
    [s->encoder setArgumentBuffer:s->arguments offset:0];
    for (size_t i = 0; i < s->count; ++i)
      [s->encoder setBuffer:c->buffers[bindings[i].buffer] offset:bindings[i].offset atIndex:i];
    memcpy(s->bindings, bindings, sizeof(view) * s->count); s->bound = 1;
  }}
  memcpy(s->constants.contents, constants, s->constant_count * 4);
  [c->encoder setComputePipelineState:c->pipelines[ordinal]];
  [c->encoder setArgumentTable:s->table];
  [c->encoder dispatchThreadgroups:groups threadsPerThreadgroup:c->threads[ordinal]];
}}
{chr(10).join(functions)}
static int setup({n}_context* c) {{
  NSError* error = nil;
  c->device = keep(c, MTLCreateSystemDefaultDevice());
  if (!c->device || ![c->device supportsFamily:MTLGPUFamilyMetal4]) return fail("Metal 4 device required");
  c->queue = keep(c, [c->device newMTL4CommandQueue]);
  c->allocator = keep(c, [c->device newCommandAllocator]);
  c->command = keep(c, [c->device newCommandBuffer]);
  c->event = keep(c, [c->device newSharedEvent]);
  c->feedback = keep(c, dispatch_semaphore_create(0));
  MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
  c->residency = keep(c, [c->device newResidencySetWithDescriptor:desc error:&error]);
  if (!c->queue || !c->allocator || !c->command || !c->event || !c->residency) return fail("creating Metal context");
  {chr(10).join(setup)}
}}
int {n}_create({n}_context** context) {{ @autoreleasepool {{
  if (!context) return fail("null context pointer");
  *context = NULL;
  if (@available(macOS 26.0, *)) {{
    {n}_context* c = calloc(1, sizeof(*c));
    if (!c) return fail("allocating context");
    NSMutableArray* owned = [NSMutableArray new];
    c->owner = (__bridge_retained void*)owned; c->owned = owned;
    if (setup(c)) {{ {n}_destroy(c); return 1; }}
    *context = c; return 0;
  }}
  return fail("Metal backend requires macOS 26 or later");
}} }}
{signature} {{ @autoreleasepool {{
  if ({nulls}) return fail("null argument");
  {n}_context* c = context;
  if (await_device(c)) return 1;
  view inputs[{max(1, ins)}] = {{{", ".join(stages) or "{0}"}}};
  view outputs[{max(1, outs)}] = {{0}}, results[{max(1, outs)}] = {{0}};
  {chr(10).join(f"memcpy(contents(c, inputs[{i}]), in{i}, {sizes[i]});" for i in range(ins))}
  if (run{main}(c, inputs, outputs, results) || await_device(c)) return 1;
  {chr(10).join(f"memcpy(out{i}, contents(c, results[{i}]), {sizes[ins + i]});" for i in range(outs))}
  return 0;
}} }}
void {n}_destroy({n}_context* c) {{ @autoreleasepool {{
  if (c) {{ await_device(c); c->encoder = nil; c->error = nil; (void)CFBridgingRelease(c->owner); free(c); }}
}} }}
'''
