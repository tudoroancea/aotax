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

## G1a: where the per-call floor goes

Scripts and raw data were written to `/tmp/aotax-metal-floor` and are not kept in the repository.

A warmed tiny dispatch took 91 µs, of which the GPU executed 7.7 µs: 5 µs to create, encode and
commit, about 40 µs of driver scheduling before the GPU starts (Metal's `kernelStartTime` and
`kernelEndTime` describe that scheduling, `GPUStartTime` and `GPUEndTime` the execution), and 36 µs
from the GPU's end to the CPU's return. A cold first call costs 3 to 4 ms; about 30 warm-up calls
remove it. A GPU kept busy by other work lowers the floor to about 41 µs, but a keep-alive is not
worth shipping.

What a host controls, measured per call on the N=12 Hessian (wall time, with staging):

| Host design | µs |
|---|---:|
| G0: compute encoder plus blit encoders for copies and fills | ~600 |
| Copies and fills as compute work in one serial encoder | 346 |
| Metal 4, whose compute encoder copies and fills natively | 328 |
| Metal 4 with the whole call pre-encoded in an indirect command buffer | 326 to 352 |

Encoder switches were the main cost: encoding the N=12 call fell from about 139 µs to 24 µs with
one encoder, and to 5-10 µs pre-encoded. Spinning on a shared event instead of blocking saves 13 to
20 µs but occupies a CPU core. Private storage, pipeline hints, unretained command buffers,
concurrent encoders and untracked resources made no consistent difference. Metal's API validation
roughly doubles the latency and shader validation triples it.

With the best configuration the batch crossover moves to about 48 Hessians at N=12 (2.4× at 64,
5.5× at 128) and about 12 at N=50 (3.0× at 32, 7.2× at 128). Single evaluations stay 30× slower than
the CPU at N=12 and 4× slower at N=200.

Decision for G1b: Metal 4 (macOS 26 or later), one queue and reusable command-buffer object per
context, context-owned shared buffers, cached pipelines, argument tables and a residency set,
dispatches and native copies and fills in one compute encoder with the HAL program's barriers, and a
blocking wait on a shared event. Pre-encoding the whole call is a later option.

## G1b: production backend

`aotax.compile(..., device="metal")` now generates the same C interface from the shared HAL
translator. `host.py` keeps one implementation of command recording, control flow, alias analysis
and allocation liveness. Its device hooks use buffer-plus-offset values on Metal. The 399-line
`metal.py` reads IREE 3.12's `MTL1` executable schema and generates an ARC Objective-C host.
Each context owns shared storage, cached pipelines and argument tables, a residency set, a Metal 4
queue, command allocator and reusable command buffer. The host re-encodes one compute encoder per
queue execution and preserves HAL barriers. It stages caller inputs and outputs with `memcpy`.
Overlapping native copies use scratch storage; non-byte fill patterns expand a short seed through
native copies. Host branches and loops remain host control flow. Every iteration that executes
GPU work pays submission and completion latency, so small iterative programs are a poor fit.

Fences and host reads block on a shared event. The host also waits for commit feedback so device
errors reach the C return value and stderr before storage is reused or destroyed. This adds
completion work compared with G1a's timing prototype, which did not check feedback on every call.
MSL compiles at `create`; the optional Xcode Metal toolchain is not needed. Building the host does
require a macOS 26 or later SDK. `create` rejects older systems and devices without Metal 4.
The API accepts only float32 and int32 arrays. Float64 intermediates and custom calls are rejected;
IREE-supported internal integer indices retain their original width, including int64 indices
from JAX/asdex. Neither BLAS routing nor the float64 exponential rewrite runs on Metal.

Same M4 Max and macOS 27.0.1 as G0/G1a. Times below are wall microseconds, including caller staging,
best of five fresh processes per case. Each process alternates CPU/Metal timing order by pass,
warms each backend 30 times, checks every output against JAX float32 with
`abs(error) <= 3e-4 * (1 + abs(reference))`, then times 100 Metal calls and about 20 ms of CPU
calls in C. No Python-call overhead or background GPU workload is included. The machine is shared;
GPU power state and other work cause large variation. CPU and Metal use the same float32 function.
Single evaluations and `jax.vmap` at batch one are measured separately. The maximum absolute
error across these cases was 6.08e-4 for Metal and 4.84e-4 for CPU.

| Case | CPU µs | Metal µs | CPU/Metal |
|---|---:|---:|---:|
| One tiny dispatch | 0.035 | 113.8 | 0.0003× |
| N=12 single | 11.44 | 408.4 | 0.03× |
| N=50 single | 38.26 | 350.9 | 0.11× |
| N=200 single | 111.25 | 390.7 | 0.28× |
| N=12 batch 1 | 11.42 | 414.4 | 0.03× |
| N=12 batch 2 | 23.04 | 320.8 | 0.07× |
| N=12 batch 4 | 43.46 | 335.8 | 0.13× |
| N=12 batch 8 | 92.71 | 343.6 | 0.27× |
| N=12 batch 16 | 211.46 | 374.9 | 0.56× |
| N=12 batch 32 | 420.02 | 402.7 | 1.04× |
| N=12 batch 40 | 484.76 | 439.9 | 1.10× |
| N=12 batch 48 | 627.00 | 418.4 | 1.50× |
| N=12 batch 64 | 1107.75 | 471.0 | 2.35× |
| N=12 batch 96 | 1229.75 | 547.7 | 2.25× |
| N=12 batch 128 | 3122.80 | 699.1 | 4.47× |
| N=50 batch 1 | 38.23 | 357.1 | 0.11× |
| N=50 batch 2 | 84.54 | 339.3 | 0.25× |
| N=50 batch 4 | 201.86 | 351.9 | 0.57× |
| N=50 batch 8 | 394.02 | 407.6 | 0.97× |
| N=50 batch 10 | 478.46 | 427.9 | 1.12× |
| N=50 batch 12 | 572.74 | 425.9 | 1.34× |
| N=50 batch 16 | 788.00 | 470.1 | 1.68× |
| N=50 batch 32 | 1670.55 | 686.1 | 2.43× |
| N=50 batch 64 | 3679.50 | 989.7 | 3.72× |
| N=50 batch 128 | 10400.85 | 1641.7 | 6.34× |

The first observed wins are batch 32 at N=12 and batch 10 at N=50. Those margins moved between
sweeps; batch 48 and batch 12 give clearer wins. Single evaluations still lose by 36×, 9× and
3.5× at N=12, 50 and 200. The CPU float64 benchmark remains 16.1 / 58.4 / 200.1 µs, best of three.

Reproduce with `uv run --group bench benchmarks/metal.py`. It writes standalone C timing binaries
and all raw rounds to `build/metal/results.json`. `--build-only` and `--measure-only` separate
compilation from timing; `--cases N12_single N12_B48` narrows a run. No existing benchmark harness
files changed. Tests cover elementwise work, matmul, an MLP, pytrees, mixed types, scan/while/branches,
constants, no-dispatch programs, changing inputs, concurrent contexts, aliases, native overlaps,
fills, unsupported types and MSL errors. All 18 pass with `MTL_DEBUG_LAYER=1`; the production
N=12 batch-128 timing binary also passes API validation. The full suite passes with 98 tests and
one expected IREE overlap xfail. Ruff lint/format, ty and the required CPU benchmark pass.

The CPU's known overlapping `dynamic_update_slice` example does not compile on Metal with IREE
3.12: its read/write `hal.interface.binding.subspan` fails Metal legalization. The host fixes
overlapping native HAL copies, but does not repair IREE's unsupported dispatch lowering.

The requested cross-family review was launched with Claude Fable 5.1 at high effort, but failed
with "Claude API rate limit reached. Try again later." No review verdict or delta check exists.
