# Phase 2: more workloads, a Metal path, solvers and control flow

Phase 1 (commit 1af8fad) made the npmpc Hessian 1.9 / 1.6 / 1.6× faster through a generated C
host, Accelerate for matrix products and a vectorized `exp`; AGENTS.md records what was learned.
Everything so far was tuned on one workload dominated by matrix products. This phase widens what is
measured and what aotax can compile.

Sources: Scaly's suite on `main` (`/Users/tudoroancea/dev/scaly/benchmarks`, `harness/sweep.py`:
chain, race_cars, unbumpercars, npmpc, npmpc_decoder, each Hessian and `_jac`), and Scaly's
`devrush` branch (`git -C /Users/tudoroancea/dev/scaly show origin/devrush:<path>`): generated
interior-point solvers (`src/scaly/opt/ipm`, stagewise KKT), Newton and implicit-function solvers
(`src/scaly/roots`), exports to CasADi and acados (`src/scaly/export`), and its `bench/` tree.

## Review of phase 1

- [x] Cross-family review of 1af8fad (Fable 5.1, xhigh), fixes, and delta check (Fable 5.1, low): all eight findings resolved.
- [x] Fix a silent input mix-up when `jax.export` drops an unused argument (checked by gpt-6-astra); push.

## F. Scaly's workloads

- [x] F0. Multi-workload harness (`benchmarks/README.md`): workloads discovered from
  `benchmarks/workloads/*.py`, generic C++ check and timing, `bench.py --workloads`. npmpc_jac
  added: 9.1 / 33.0 / 126.3 µs against Scaly's 9.7 / 40.9 / 165.5 (0.94× / 0.81× / 0.76×).
- [ ] F1. One agent per problem adds its module(s) only: chain (+jac), race_cars (+jac),
  unbumpercars, npmpc_decoder (+jac). Parallel after F0.
- [ ] F2. Sweep all workloads, profile the worst ratios, choose the next optimization round.

## G. Metal GPU path

Apple GPUs have no f64 arithmetic, so this path is `float32` only, with f32 tolerances.

- [ ] G0. Spike: compile f32 functions with IREE's Metal target, read the HAL program and the
  executable format (MSL or metallib, entry points, threadgroup sizes, binding and constant
  layout), run them from a hand-written Metal host, and measure the per-call floor (encode, commit,
  wait) against the CPU path over sizes. Also measure IREE's own Metal runtime driver as a reference.
- [ ] G1. If G0 shows a size range where the GPU wins: `aotax.compile(..., device="metal")`, with the
  same C API, generated from the same HAL translation (dispatches encoded into one command buffer,
  barriers as memory barriers, transients as shared buffers).
- [ ] G2. Add f32 GPU columns to the benchmark sweep and map where CPU or GPU wins.

## H. Beyond Hessians: control flow, solver loops, foreign calls

`jnp.linalg.solve` and `jnp.linalg.cholesky` do not compile today: JAX lowers them to LAPACK
custom calls on CPU, which IREE rejects. Foreign function interface (FFI) calls through
`jax.ffi.ffi_call` are custom calls too. Supporting custom calls as calls from the generated host
into C (LAPACK from Accelerate, or a user's solver) serves both the optimization loops and FFI.

- [ ] H0. Design spike: survey `devrush` (roots, opt/ipm, export, bench) and pick representative
  workloads (a Newton or implicit solve with `while_loop`, a `scan` rollout, a fixed-iteration
  interior-point or SQP loop); design custom-call support (how a `stablehlo.custom_call` reaches
  the HAL program as a host call, buffer passing, placement inside loops); write a design note.
- [ ] H1. Custom calls: LAPACK solves and Cholesky, and user FFI targets.
- [ ] H2. The H0 workloads in the benchmark harness, against Scaly's generated solvers where an
  equivalent exists and against JAX's own XLA CPU execution.

## Order

Review, F0, G0 and H0 start together; F1 follows F0; G1 follows G0; H1 and H2 follow H0. The
reviewer only reads. F agents own `benchmarks/`; G and H spikes work outside the repository or in
new modules until their design is settled.
