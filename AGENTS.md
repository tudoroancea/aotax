# aotax

## Layout

- `src/aotax/__init__.py`: `compile` exports StableHLO with `jax.export`, runs `iree-compile` to
  the HAL phase, generates `host.c`, and archives it with IREE's dispatch object.
- `src/aotax/host.py`: translates IREE's HAL program into a plain C host. No IREE runtime.
- `src/aotax/abi.h`: IREE's executable library ABI v0 (library version 6), as of
  `iree-base-compiler==3.12.0`. Bump the pin, `abi.h` and the version check together.
- `src/aotax/blas.py`: on macOS, pure contraction dispatches become Accelerate `cblas_*gemm` calls.
- `src/aotax/exp.py`: rewrites f64 `stablehlo.exponential` into a vectorizable polynomial.
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
- Custom calls keep tensor results until HAL; exporting empty destination buffers can reorder reads before calls.
  Registered adapters borrow inputs, return fresh context buffers, and keep LAPACK `info` as tensor data.
- Loop result slots avoid live pointers and call operands, including operands that die at the call.
- User FFI targets require `compile(foreign=...)`; descriptors carry static tensors, `Foreign` requests context scratch, and `load(extra=...)` links implementations.

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
