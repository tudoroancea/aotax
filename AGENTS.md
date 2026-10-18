# aotax

## Layout

- `src/aotax/__init__.py`: `compile` exports StableHLO with `jax.export`, runs `iree-compile` to
  the HAL phase, generates `host.c`, and archives it with IREE's dispatch object.
- `src/aotax/host.py`: translates IREE's HAL program into a plain C host. No IREE runtime.
- `src/aotax/metal.py`: Metal 4 device hooks for that translator and IREE 3.12's `MTL1` FlatBuffer schema; emits ARC Objective-C.
- `src/aotax/abi.h`: IREE's executable library ABI v0 (library version 6), as of
  `iree-base-compiler==3.12.0`. Bump the pin, `abi.h` and the version check together.
- `src/aotax/blas.py`: on macOS, pure contraction dispatches become Accelerate `cblas_*gemm` calls.
- `src/aotax/math.py`: expands f64 `exp` and separates broadcasted libm results from their consumers
  in programs without dot operations.
- `src/aotax/custom_calls.py`: preserves tensor custom calls through HAL and generates Accelerate LAPACK adapters.
- `benchmarks/`: the npmpc Lagrangian Hessian from Scaly, timed against Scaly's generated C.

## Checks

```bash
uv run pytest
uv run ruff check && uv run ruff format --check
uv run --group bench ty check
uv run --group bench benchmarks/bench.py --reps 5
```

## Host invariants

- Compiled with `--iree-hal-memoization=false`: each command buffer is recorded and executed in one
  block, so its commands run at the execute point. Other patterns are rejected at compile time.
- Allocation sizes must be compile-time constants; transients live in the context. Loop-carried
  allocations get a pool of (live buffers + 1) slots.
- Flush-to-zero is set once per call, as IREE's runtime did per dispatch, and the caller's mode is
  restored. The polynomial `exp` explicitly selects zero below the smallest normal to match IREE.
- Input pointers bind directly when 64-byte aligned, not overlapping outputs, and never written
  by the HAL program; otherwise they are staged through context buffers. Overlapping outputs are staged too.
- IREE 3.12's `slow_memcpy` dispatch mishandles overlapping self-slices in `dynamic_update_slice`.
- IREE 3.12's f32 `sin`/`cos` on CPU and Metal lose range-reduction accuracy around `|x| = 1e3` (absolute error ~3e-5, growing with magnitude).
- Custom calls keep tensor results until HAL; exporting empty destination buffers can reorder reads before calls.
  Registered adapters borrow inputs, return fresh context buffers, and keep LAPACK `info` as tensor data.
- Loop result slots avoid live pointers and call operands, including operands that die at the call.
- User FFI targets require `compile(foreign=...)`; descriptors carry static tensors, `Foreign` requests context scratch, and `load(extra=...)` links implementations.
- HAL buffer subspans are pointer offsets; optimization barriers are identities in the C host.
  Both retain their input buffers' aliases for liveness and input staging.
- Metal uses buffer-plus-offset values with the same HAL analysis, liveness and control flow as CPU.
  One reusable Metal 4 command buffer is re-encoded per queue execution; its allocator and argument
  buffers are reused only after completion. One compute encoder preserves HAL barriers and orders
  native copies/fills. Overlapping native copies use scratch; non-byte fills expand a seed with copies.
- Metal stages all caller arrays through context-owned shared buffers. Fences and host reads wait on
  a shared event and check commit feedback. Pipelines, argument tables and residency live in the context.
  It requires macOS 26 and a Metal 4 device. It rejects float64 and custom calls, preserves internal
  integer widths, and skips CPU BLAS routing and the f64 exp rewrite.

## Measuring

- The benchmark binaries check every entry against a dense reference before timing.
- The machine is shared and noisy. Compare candidates in one interleaved run of fresh processes,
  best of five, and judge by the ratio to Scaly or to an interleaved baseline.
- Profile with `sample <pid> 4` on a running benchmark; ignore `benchmark::CPUInfo::CPUInfo()`,
  which is Google Benchmark's startup.

## Tried and rejected (npmpc Hessian, M4 Max, iree-base-compiler 3.12.0)

- Compiler flags: `--iree-opt-level=O3` and aggressive fusion fail to compile, or give wrong
  Hessians with distribution disabled; transpose propagation, horizontal contraction fusion and
  matmul generalization are slower; data tiling fails at run time; vector widths of 32 or 64 bytes
  are 1.2 to 1.9× slower; ukernels, binding optimization and reshape movement change nothing.
- IREE's SME code for f64 matmuls is 18 to 60× slower than its NEON code, or fails to compile.
- `--iree-execution-model=inline-static` fails in codegen; `--vector-library` is not an
  `iree-compile` option.
- Optimization barriers around dot operands or results, to send more matmuls to Accelerate, and a
  zero-matrix strength reduction: slower at one horizon or more.
- The polynomial `exp` loses when range reduction emulates round-to-even; it wins with
  round-away-from-zero (`frinta`).
- Dispatches fusing a matmul with a larger broadcast consumer do not recompute the matmul. They are
  slow because IREE generates scalar code with register spills for them.

## Tried and rejected (sparse dynamics, same machine and compiler)

- Disabling vector peeling slows chain_jac M=3 by 17%, chain M=17 by 22%, npmpc N=200 by 7%
  and the width-16 decoder by 10% in an interleaved comparison.
- A vectorized f64 `tanh`, with a [9/10] Padé approximation on [-1, 1] and `exp` outside it,
  stays within 3 units in the last place of libm on 1.1 million samples, including special values. It improves the
  race-car Jacobians by only 1-3% and barely changes the Hessians; not kept.
- Barriers after every f64 `sin`, `cos` and `tanh` slow npmpc N=200 by 12%. Keep matrix-product
  programs' fusion; separate only libm results that feed broadcasts in programs without dot operations.
  Even the narrower broadcast rule slows unbumpercars at 8 cars by 7% if applied across products.
- Sparse differentiation dominates the remaining gap: Scaly and asdex agree on the nonzeros,
  but asdex's color batches do 4.5-11.5 times the estimated scalar arithmetic of Scaly's program
  on the four profiled cases. Host work outside dispatches and memory operations is only
  0.9-2.8 µs per call. See PLAN.md, F2.
