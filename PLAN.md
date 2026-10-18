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

## Reviews

- [x] G1b (Metal), F2 and the merge: gpt-6.1-sol (xhigh) found Metal's default fast math breaking
  NaN, infinity and zero semantics; fixed with `MTLMathModeSafe` and nonfinite tests, checked by
  gpt-6.1-sol (high). IREE's float32 `sin`/`cos` lose accuracy beyond |x| ≈ 1e3 on CPU and Metal;
  documented as a limit. H1c and H2 delta checks: resolved (Fable 5.1, low).

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
- [x] F2. The losses on chain and race_cars are mostly asdex's algorithm, not the compiler: asdex
  performs 4.5-11.5× the arithmetic of Scaly's symbolic derivatives (14 to 42 products per
  evaluation over dense color batches), and XLA running the same asdex functions is also 5-9× slower
  than Scaly. Host work outside dispatches is 1-3 µs. Kept: `src/aotax/math.py` computes broadcast
  f64 `sin`, `cos` and `tanh` before broadcasting in programs without dot operations (race-car
  Jacobians 12-13% faster; other kernels unchanged). Rejected: vector peeling off, a polynomial
  `tanh` (≤ 3 ulp, 0-3% gain), barriers after every transcendental. Next: derivatives of local
  stages assembled into sparse blocks, and sparse differentiation that skips zero tangents.

## G. Metal GPU path

Apple GPUs have no f64 arithmetic, so this path is `float32` only, with f32 tolerances.

- [x] G0. Spike: notes/metal.md. A direct Objective-C host runs IREE's MSL kernels, 1.2-2×
  faster than IREE's Metal runtime. No single evaluation wins on the GPU (one tiny dispatch costs
  ~110 µs); batches do, from ~64 npmpc Hessians at N=12 or ~16 at N=50 (3.4× and 5.4× at 128).
- [x] G1a. The floor is driver scheduling and completion latency (GPU work is 7.7 of 91 µs). One
  compute encoder (Metal 4: native copies and fills) halves the N=12 Hessian to ~330 µs and moves
  the crossover to ~48 Hessians at N=12 and ~12 at N=50. notes/metal.md has the design decision.
- [x] G1b. `aotax.compile(..., device="metal")` preserves the C API and shares HAL analysis,
  liveness and control-flow translation with CPU. A 399-line Metal module reads IREE's executable
  schema and emits an ARC Objective-C host with cached pipelines, shared staging, a residency set,
  one encoder per queue execution and blocking completion. All 18 Metal tests pass with API
  validation; CPU remains 16.1 / 58.4 / 200.1 µs. Clear batch wins start at 48 for N=12 and 12 for
  N=50, reaching 4.47× / 6.34× at 128. notes/metal.md records the full table. The requested Fable
  5.1 high-effort review could not run because the Claude API rate limit was reached.
- [x] G1c. Metal uses Safe math, with regression coverage for nonfinite values and signed zero.
  Batch-128 latency rose by less than 1%. Documented IREE's f32 trig range limit; float64 promotion
  fixes CPU accuracy but costs about 7× on an elementwise probe. notes/metal.md has measurements.
  All checks pass, including 19 Metal tests with API validation; CPU is 16.4 / 58.3 / 211.1 µs.
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
- [x] H2. `ProgramWorkload` in the harness: chain_rollout, newton, gauss_newton, chain_qp and
  batched_newton, checked against `jax.jit` (exact iteration counts), timed natively and as XLA
  amortized inside one jitted loop. aotax beats XLA on the rollout (1.0-2.3×), Gauss-Newton
  (2.2-2.9×) and the QP (1.1-1.5×), and loses on Newton for n ≥ 8 (up to 1.41× at n=32), where
  hundreds of small dispatches surround each LU. `benchmarks/scaly_devrush.py` compares with
  Scaly's devrush solvers: Newton 0.18 vs 1.20 µs at n=4 and 59.8 vs 63.0 at n=64 (same iterations);
  the QP 4.6 vs 21 µs (different algorithms and stopping rules).

## Order

Review, F0, G0 and H0 start together; F1 follows F0; G1 follows G0; H1 and H2 follow H0. The
reviewer only reads. F agents own `benchmarks/`; G and H spikes work outside the repository or in
new modules until their design is settled.
