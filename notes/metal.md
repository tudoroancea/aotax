# G0: a Metal GPU path

Findings of the G0 spike (October 2026, Apple M4 Max with 40 GPU cores, macOS 27.0.1,
`iree-base-compiler==3.12.0`). Prototypes, inputs and raw timings were written to
`/tmp/aotax-metal` and are not kept in the repository.

## What IREE produces

- `--iree-hal-target-device=metal` compiles through the `metal-spirv` backend. The default
  `--iree-metal-compile-to-metallib=true` needs Xcode's optional Metal toolchain (`metal`,
  `metallib`); with `=false` the executables carry Metal Shading Language (MSL) source, which
  Metal compiles at run time.
- Each executable is a FlatBuffer (identifier `MTL1`, format `metal-msl-fb`) after IREE's 64-byte
  executable header, with the schema in IREE's `runtime/src/iree/schemas/metal_executable_def.fbs`.
  It holds the MSL, kernel names (which can differ from the HAL export names), threadgroup sizes,
  constant counts and binding flags. Unlike the CPU path, each dispatch is its own executable.
- The HAL program uses the same operations as on the CPU: allocations, command buffers, dispatches,
  copies, fills, barriers, queue execution and fences, with the same memory types. On the CPU the
  host runs them synchronously and ignores barriers; on Metal, dependencies must be kept.
- Kernel interface, as in IREE's `hal/drivers/metal/direct_command_buffer.m`: an argument buffer at
  Metal buffer index 0 holds binding `n` at `[[id(n)]]` (set with `MTLArgumentEncoder` and an
  offset); push constants are consecutive 32-bit words at buffer index 3; resources referenced
  through the argument buffer need `useResource`; HAL workgroup counts and the executable's
  threadgroup size feed `dispatchThreadgroups`.

## A direct host works

A prototype (`hal_schedule.py`, `metal_host.m`) read the HAL program with IREE's compiler bindings,
extracted the MSL and ran the schedule from Objective-C: shared buffers owned by the context, cached
pipelines and argument buffers, one command buffer per call, compute dispatches in one encoder
until a copy or fill needs a blit encoder, HAL barriers as memory barriers. All 37 cases matched JAX
in float32, including the npmpc Hessian (80 dispatches, 16 copies, 18 fills and 61 barriers per
call), also with Metal's API and shader validation on.

## Latency

Apple GPUs have no float64 arithmetic, so this path is float32 only. All times in µs, best of five
fresh processes; "staged" includes copying inputs in and outputs out.

| | CPU (aotax) | Metal, staged |
|---|---:|---:|
| Empty command buffer, commit and wait | | 14.8 |
| One tiny dispatch | 0.4 | 118.8 |
| Elementwise `sin(x)*x + 2`, 1,048,576 elements | 398.6 | 266.4 |
| Matmul 2048×2048 | 4,995.7 | 4,845.6 |
| MLP 9→32→32→2, batch 8,192 | 492.8 | 199.8 |
| npmpc Hessian N=12 | 11.4 | 635.5 |
| npmpc Hessian N=200 | 110.5 | 641.2 |
| npmpc Hessian N=12, batch 64 | 1,111.3 | 745.6 |
| npmpc Hessian N=12, batch 128 | 3,080.3 | 908.0 |
| npmpc Hessian N=50, batch 128 | 10,346.0 | 1,909.6 |

Single evaluations never win on the GPU. Batches do, from about 64 Hessians at N=12 or 16 at N=50.
IREE's own runtime with its Metal driver (from the `iree-base-runtime` wheel) was 1.2 to 2× slower
than the direct host on the same kernels. The 100 µs between an empty command buffer and one tiny
dispatch is unexplained.

Float32 Hessians at the benchmark's sample stayed within 8e-4 of the float64 reference, on entries up
to 1,742; JAX in float32 is similar. Solver convergence in float32 is untested.

## Recommendation

Build an opt-in float32 Metal backend with the existing synchronous C API, aimed at batches; keep
the CPU path as the default. Stage caller arrays through shared buffers at first: wrapping caller
memory with `newBufferWithBytesNoCopy` requires page alignment. Estimated 500 to 900 lines with
reuse of the HAL translation, which needs buffer-plus-offset values and dependency-preserving
command emission.
