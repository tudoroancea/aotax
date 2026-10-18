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
- [x] F1. Ported chain, race_cars, unbumpercars and npmpc_decoder (Hessian and Jacobian each),
  cross-checked against Scaly's formulations. aotax / Scaly CPU time, from each port's own
  interleaved run:

  | Workload | sizes | Hessian | Jacobian |
  |---|---|---|---|
  | npmpc | N = 12, 50, 200 | 1.06×, 0.94×, 0.81× | 0.94×, 0.81×, 0.76× |
  | npmpc_decoder | width 16, 64, 256 | 1.68×, 0.47×, 0.18× | 1.05×, 0.58×, 0.20× |
  | unbumpercars | 2, 4, 8 cars | 0.31×, 0.31×, 0.42× | 0.62×, 0.35×, 0.40× |
  | chain | M = 3, 9, 17 | 3.83×, 3.09×, 1.99× | 16.5×, 5.78×, 4.23× |
  | race_cars | N = 5, 40, 500 | 5.40×, 4.56×, 5.24× | 5.85×, 5.48×, 5.68× |

  aotax wins where neural networks dominate (Accelerate, vectorized code) and loses 2-16× on small
  dense dynamics without matrix products, where time spreads over elementwise and transpose
  dispatches, copies, fills and scalar `sin`/`cos`/`tanh` calls.
- [x] F2. Sweep all workloads, profile the worst ratios, choose the next optimization round.
  - [x] Record one full interleaved baseline, three fresh processes per binary.
  - [x] Measure coloring, arithmetic, Accelerated Linear Algebra (XLA) loop execution and dispatch costs for chain_jac M=3,
    chain M=17 and both race_cars derivatives at N=500. This depends on baseline compilation.
  - [x] Test general optimizations identified by the measurements, keeping host edits isolated
    from H1. Record rejected candidates and compare retained changes in a full sweep.
  - [x] Run tests, lint, format and types; document results and attempt the Claude review.
    Commit locally after validation. H1 and the Metal investigation can proceed independently.
  - The four cases use 14 reverse products, 42 Hessian-vector products, 4 Hessian-vector products,
    and 6 reverse products respectively. Scaly and asdex agree on the sparse coordinates.
    Estimated scalar arithmetic is 0.49 / 19.58 / 4.27 / 2.62 million operations for asdex/XLA,
    versus 0.068 / 4.35 / 0.558 / 0.228 million for Scaly's optimized scalar program.
    These are compiler estimates before machine-code generation, with loops and calls counted.
  - XLA's loop wall times are 36 / 1153 / 354 / 211 µs, with all inputs perturbed each iteration
    and all output entries consumed. XLA uses extra CPU threads. Disabling its parallel task
    assignment gives 37 / 2168 / 519 / 359 µs; aotax takes 64 / 2441 / 256 / 231 µs on one thread.
    Scaly takes 3.92 / 1145 / 49.02 / 40.82 µs. The main gap is sparse differentiation and the
    dense intermediate arrays. Host work outside dispatches and memory operations takes
    0.9 / 2.3 / 2.8 / 1.4 µs; replacing dispatch kernels with no-ops gives 1.1 / 2.5 / 3.2 / 1.5 µs.
    The chain Hessian fills 29 MB per call; race-car transpose dispatches take about 55 µs.
  - Selected broadcast barriers for f64 `sin`, `cos` and `tanh` in programs without dot operations.
    An interleaved best-of-five comparison cuts race-car Jacobians by 12-13%, with unchanged
    Hessian times. Matrix-product programs need the exclusion: the same broadcast barriers slow
    unbumpercars at 8 cars by 7%. Tests: 42 passed, one expected IREE failure. Lint, format and
    types pass. Both full sweeps, three interleaved processes per binary, pass every dense check.
    Claude Fable 5.1 at high effort could not review dd21ee4: its API rate limit was reached.
    The exclusion also covers scalar dot operations in race-car Hessians. Only the three
    race-car Jacobian kernels change; host and dispatch objects are identical for the other 27 cases.
    Full-sweep race-car Jacobians improve from 2.67 / 16.67 / 230.98 to 2.30 / 14.61 / 200.17 µs
    at N=5 / 40 / 500. At N=500, dispatches drop from 114 to 109 and scalar libm results from
    21,500 to 11,500, counting a paired sin/cos call as two results. The remaining work belongs
    mostly to dense color batches and their copies, fills and transposes. Next: local stage and
    link derivatives with sparse block assembly, sparse automatic differentiation that skips
    zero tangents, and direct lower-triangle decompression in the JAX formulation or asdex.

## G. Metal GPU path

Apple GPUs have no f64 arithmetic, so this path is `float32` only, with f32 tolerances.

- [x] G0. Spike: notes/metal.md. A direct Objective-C host runs IREE's MSL kernels, 1.2-2×
  faster than IREE's Metal runtime. No single evaluation wins on the GPU (one tiny dispatch costs
  ~110 µs); batches do, from ~64 npmpc Hessians at N=12 or ~16 at N=50 (3.4× and 5.4× at 128).
- [x] G1a. The floor is driver scheduling and completion latency (GPU work is 7.7 of 91 µs). One
  compute encoder (Metal 4: native copies and fills) halves the N=12 Hessian to ~330 µs and moves
  the crossover to ~48 Hessians at N=12 and ~12 at N=50. notes/metal.md has the design decision.
- [ ] G1b. After H1a+b lands: `aotax.compile(..., device="metal")`, with the
  same C API, generated from the same HAL translation (dispatches encoded into one command buffer,
  barriers as memory barriers, transients as shared buffers).
  - [x] Share HAL analysis and control flow; add Metal executable extraction and command emission.
  - [x] Wire compilation and loading, then test control flow, staging and Metal API validation.
  - [ ] Measure the production backend, update docs, run checks and cross-family review, commit.
- [ ] G2. Add f32 GPU columns to the benchmark sweep and map where CPU or GPU wins.

## H. Beyond Hessians: control flow, solver loops, foreign calls

`jnp.linalg.solve` and `jnp.linalg.cholesky` do not compile today: JAX lowers them to LAPACK
custom calls on CPU, which IREE rejects. Foreign function interface (FFI) calls through
`jax.ffi.ffi_call` are custom calls too. Supporting custom calls as calls from the generated host
into C (LAPACK from Accelerate, or a user's solver) serves both the optimization loops and FFI.

- [x] H0. Design spike: notes/custom-calls.md. Custom calls become private tensor `func.call`s,
  which IREE keeps as HAL `util.call`s; the host calls C adapters. Five workloads prototyped (chain
  rollout, Newton, Gauss-Newton fit, chain QP interior point, batched Newton); only the rollout
  compiles today. Scaly's generated Newton is 0.5 µs against aotax's 2.0 µs: dispatch count
  dominates small control-flow programs.
- [x] H1a+b. `src/aotax/custom_calls.py`: JAX's LAPACK getrf, potrf and trsm custom calls (f32,
  f64, batched, inside loops) become synchronous calls into Accelerate with context scratch, so
  unchanged `jnp.linalg.solve`, `cholesky` and `solve_triangular` compile. A solve takes 0.31 µs at
  n=4 and 15.6 µs at n=64; the five H0 workloads run 2-4× faster than JAX called from Python.
  Reviewed by Opus 5.5 (xhigh), fixes checked by Fable 5.1.
- [x] H1c. User FFI targets: explicit `foreign` registry, tensor descriptors, optional context scratch,
  selective export exemptions and `load(extra=...)` linking. Full checks pass. Scaly Newton inside
  an eight-solve continuation loop takes 4.82 µs against direct Scaly's 4.59 µs, agrees with JAX
  to 2.8e-16; `/tmp/aotax-h1c/scaly_foreign.py`. Fable 5.1 high-effort review approved with no
  act-on findings. IREE 3.12's scalar-result loop crash is documented; vector statistics work.
  Two small review notes addressed; local checks pass. Delta review blocked by Claude's rate limit.
- [ ] H2. The H0 workloads in the benchmark harness, against Scaly's generated solvers where an
  equivalent exists and against JAX's own XLA CPU execution.

## Order

Review, F0, G0 and H0 start together; F1 follows F0; G1 follows G0; H1 and H2 follow H0. The
reviewer only reads. F agents own `benchmarks/`; G and H spikes work outside the repository or in
new modules until their design is settled.
