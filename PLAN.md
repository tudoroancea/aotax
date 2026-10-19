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
  N=50, reaching 4.47× / 6.34× at 128. notes/metal.md records the full table. Fable 5.1 at high
  effort approved the change after the rate-limited review resumed; no actionable findings.
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
  Two small review notes addressed; local checks pass. Fable 5.1's resumed low-effort delta check
  confirmed both fixes, with nothing open.
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

## Phase 3

- [x] I. `compile(approximate_math=False)` gives libm-quality elementary functions by default. CPU
  keeps f64 `sin`/`cos`/`exp`/`log`/`tanh` in dispatches and promotes f32 ones to f64; other
  functions call generated libm adapters, and JAX's `erfc`/`exp2`/`log2` are preserved
  before export. The default keeps subnormals; opt-in flushes them (as XLA's CPU backend does),
  expands f64 `exp` and keeps IREE's approximations, which no 3.12 option disables. Dense sweeps
  of 24 functions stay within 2 ulp. Opt-in is 2-6% faster on npmpc, equal on race cars, 3.6× on
  an f32 `sin` probe. Shortcut: Metal's default runs elementary functions on the host after
  waiting for the GPU, 3-5× slower on npmpc; accurate f32 GPU implementations would remove it.
  Reviewed by gpt-6.1-sol (xhigh): stale constants on recompiling, float16/complex intermediates,
  floating-point mode left changed by foreign code, and reserved target and symbol names; all
  fixed with regression tests and confirmed by gpt-6.1-sol (high).
- [x] J0. Structured derivatives spike (notes/structured-derivatives.md). Prototypes differentiate
  small local programs with JAX and assemble only the needed entries into static coordinates. With
  the outer Lagrangian factored by hand, CPU µs asdex / structured / Scaly: race_cars N=500 Hessian
  253 / 55 / 49 and Jacobian 205 / 40 / 41; npmpc N=200 Hessian 200 / 174 / 249; unbumpercars 8
  Hessian 245 / 61 / 575; chain_jac M=17 1665 / 183 / 401. The chain Hessian at M=9 and 17 stays at
  asdex's speed, 2-3× slower than Scaly. Automatic extraction worked only on small whole programs.
- [x] J1. `aotax.sparse`: `sparse_hessian` (lower triangle) and `sparse_jacobian` of one selected
  float argument return a values function, rows, cols, shape and a report. `method="auto"` extracts
  local structure and falls back to asdex, `"structured"` raises instead of falling back, `"asdex"`
  is the baseline with direct lower-triangle recovery. asdex is an optional extra. Workloads supply
  their plain JAX function; the benchmarks build both columns through `aotax.sparse`.
- [x] J2. Automatic extraction from unchanged workload functions: one local program per instance of
  a repeated computation over static windows or gathers, Hessians as `jax.hessian` of grouped local
  primal sums, Jacobians in forward or reverse mode by width, shared parameters unbatched, scalar
  leaves for small programs without products, static gather-and-add assembly. A first version that
  split outputs and took Jacobians of `jax.grad` neighborhoods was up to 2.5× slower than asdex.
  µs asdex / auto / Scaly: race_cars N=500 205 / 70 / 48, race_cars_jac 172 / 42 / 40, npmpc
  N=200 204 / 150 / 249, npmpc_jac 132 / 80 / 163, decoder width 256 221 / 145 / 1167, chain_jac
  M=17 1648 / 204 / 396. The chain Hessian and unbumpercars fall back (J3, J4). Construction takes
  under 2 s, 7-11 s for chain_jac M=9/17. Reviewed by gpt-6.1-sol (xhigh): constant dtypes in
  scalar replay, filled gathers read as coordinates, mixed parameter dtypes, recursion depth,
  control-flow fallbacks and quadratic assembly storage fixed; a fix check (high) found nested
  bodies missed by the asdex out-of-bounds guard and an over-broad `cond` rejection, both fixed.
  asdex 0.5.2 drops clamped out-of-bounds gather indices; aotax rejects them.
- [x] J3. Hessian terms with the same input neighborhood are grouped, including neighbor coupling
  repeated inside a stage; groups over the operation budget split into smaller sums. Construction
  is bounded before scalar differentiation: input width times local argument, result and operand
  sizes (custom JVP rule bodies counted two levels deep) at most 262,144, and at most 256 local
  programs per pass. µs asdex / auto / Scaly, chain Hessian: M=3 74 / 16 / 22, M=9 942 / 758 / 303,
  M=17 2162 / 1805 / 1042; other workloads within ±1% of main. Construction takes 0.9 / 6.4 / 9.6 s.
  At M=17 the pattern keeps 45,000 structural zeros beyond asdex's 73,923 entries, and native
  compilation peaks at 3.4 GiB, against 0.96 GiB for the asdex path. Pruning gradient components
  before the second differentiation made IREE compilation too heavy; reverse-over-reverse was
  slower. Reviewed by gpt-6.1-sol (xhigh): custom JVP rule bodies escaped the construction budget;
  the fix check (high) found that nested rules still did. Both fixed with regression tests; the
  second fix was confirmed with the checker's probe, without another review round.
- [x] J4. Shared local maps: a computed array read by several factors through gathers or
  overlapping slices is kept as one map. Outputs no factor uses are dropped, its local Jacobian and
  second derivatives are computed once per instance, and derivatives compose through it. µs asdex /
  auto / Scaly: unbumpercars Hessian 8 cars 234 / 53 / 563, Jacobian 68 / 30 / 167; other workloads
  unchanged. Several computed gather sources and dtype changes fall back. The Sol review stopped at
  capacity and the Opus one on API errors; Opus 5.5 (this session) finished it with adversarial
  probes and reruns of the J2 review probes. Fixed: a crash when no used output depends on the
  input, which now falls back.
- [ ] J5. `scan`, where carry analysis proves local structure; nonlinear rollouts stay dense.
- [x] K0. Selective scalarization spike (notes/scalarization.md). IREE cannot reproduce Scaly's
  scalarization: Scaly's gain comes from turning a whole local stage function into scalar code with
  shared subexpressions inside a stage loop, while asdex's graph reaches IREE as batched color and
  stage arrays with small axes inside. No IREE option, fusion policy or StableHLO pre-pass helps
  across the board. Two bounded mechanisms do: power-of-two small-dimension tiles (race_cars N=500
  16% faster) and plain C for dispatches whose tensors all have at most 256 elements (race_cars N=5
  26% faster). Neither changes chain or npmpc.
- [x] K1. CPU compilation stops at executable configurations, fills vector tile dimensions of
  extent 2, 4 or 8 in projected-permutation `linalg.generic`s (at most 32 elements per level),
  then resumes (src/aotax/cpu.py). race_cars N=40/500 16% faster, race_cars_jac N=500 13%; other
  workloads within ±3%; compilation 0.1 s slower. Reviewed by gpt-6.1-sol (high): scalable tile
  lists and scalar operands are now left to IREE.
- [ ] K2. Tiny-dispatch C emitter. Wait for J, then measure on its local stage functions, where a
  whole stage may fit the budget.

I and K1 are independent. One agent does J1 then J2; J3 and J4 follow J2 and can run in parallel
with separate file ownership; K2 follows J2.
