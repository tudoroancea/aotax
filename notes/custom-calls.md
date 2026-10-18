# H0: control flow, linear solves and foreign calls

Design note from the H0 spike (October 2026). Prototypes and raw measurements were written to `/tmp/aotax-h0` and are not kept in the repository.

Recommendation: rewrite supported Stable High Level Operations (StableHLO) custom calls to private tensor-valued `func.call`s before IREE. IREE 3.12 preserves these as hardware abstraction layer (HAL) `util.call`s with buffer-view arguments and results. Extend the generated C host to allocate the result buffers and call a C adapter directly. This works inside JAX loops and needs no IREE runtime. Start with Linear Algebra Package (LAPACK) LU, Cholesky and triangular solves, then expose the same mechanism to user-provided C adapters.

The working prototype compiles an unchanged `jnp.linalg.solve`, including inside `lax.fori_loop` and data-dependent `lax.while_loop`. Its C adapters call Accelerate's `dgetrf_`, `dpotrf_` and `cblas_dtrsm`. A separate foreign function interface (FFI) prototype also calls `dgetrf_` plus `dgetrs_` through `jax.ffi.ffi_call`. All five workload prototypes load and agree with JAX after the rewrite. Only the rollout compiles with today's aotax.

No tracked repository files were changed. No commits were made. Everything described here lives under `/tmp/aotax-h0`.

## Evidence and scope

The aotax checkout is commit `549916104494253f301569960a2f374c75b9341f`. Sources read were `AGENTS.md`, `PLAN.md`, especially H, `README.md`, and all Python modules under `src/aotax`. Scaly sources came from `origin/devrush`, commit `6aa1e6b489d77c742ac1af0afd99417733fb713e`, through `git ls-tree` and `git show`. The working Scaly checkout was never switched or edited. The extracted source copy is `scaly/src/scaly` beneath this directory.

Scaly files informing the design:

- `roots/newton.py`, `roots/least_squares.py`, `roots/implicit.py`, `roots/problem.py` and `roots/solver.py` establish generated Newton and least-squares loops, simplified Newton with reused factors, and implicit differentiation at the converged solution.
- `opt/ipm/algorithm.py`, `kkt.py`, `stages.py`, `ruiz.py` and `method.py` establish a generated PIQP-style proximal interior-point method (IPM), scaling, factorization retry and refinement loops, and dense, sparse and stagewise linear algebra. This method solves quadratic programs (QPs). A nonlinear optimal control problem (OCP) needs an outer method such as sequential quadratic programming (SQP).
- `opt/external/wrapper.py` and `model.py` establish solver adapters, oracle callbacks, caller-owned workspace and solver statistics. Scaly's generated solver adapters are useful eventual FFI targets, but do not automatically satisfy an XLA FFI calling convention.
- `export/casadi.py` and `export/acados.py` show generated oracle integration. CasADi's column-major dense matrices need conversion from aotax's row-major arrays. A full acados integration is separate work from the custom-call mechanism.
- `ocp/direct.py` and `ocp/ilqr.py` establish sparse or condensed optimal control formulations and structured solver alternatives. The Karush-Kuhn-Tucker (KKT) systems in the IPM can exploit this structure.
- `bench/problems/chain/__init__.py`, `npmpc/__init__.py`, `race_cars/__init__.py` and `bench/harness/gbench.py` establish problem sizes, dynamics, constraints and native benchmarking. Chain has 3 controls and `3*(2*(M-2)+1)` states. Its RK4 rollout is a compact initial control-flow workload. Npmpc adds neural dynamics and larger matrix products. Race cars adds nonlinear vehicle dynamics and horizon scaling.

Experiments used an Apple M4 Max, macOS 27.0.1, Python 3.14.6, JAX and jaxlib 0.11.2, NumPy 2.5.4 and `iree-base-compiler==3.12.0`. Values below use float64. This is a macOS CPU design. Linux numerical adapters would need an explicitly linked LAPACK/BLAS provider and its integer ABI; neither was tested. It makes no claim about Metal buffer access or asynchronous GPU calls.

## Representative workloads

`workloads.py` contains runnable JAX implementations of all five. JAX's Accelerated Linear Algebra (XLA) CPU compiler provides the execution baseline. They return arrays and integer iteration counts supported by the current public ABI.

| Workload | JAX program and smoke size | What it exercises | Baselines and size axes |
|---|---|---|---|
| Newton root solve | One-dimensional Bratu residual, differentiated by `jax.jacfwd`, dense `jnp.linalg.solve`, `lax.while_loop`, n=8, tolerance 1e-10, maximum 20 iterations | Jacobian evaluation, LU and triangular solves, host branches, convergence-dependent exit and iteration count | Exact Scaly `roots.Newton` mirror built and timed here; JAX/XLA CPU. Sweep n=4..64 initially, tolerance, initial guess and Bratu parameter. Scaly's existing 2D Bratu example suggests a later sparse extension, rather than dense n-squared scaling. |
| Chain rollout | Scaly's chain dynamics and fourth-order Runge-Kutta step in `lax.scan`, M=3, horizon 12 | Serial loop carry, many small dispatches, reductions, dynamic slices and full trajectory output; no custom calls | JAX/XLA CPU; construct the exact Scaly `scan` of `chain_step_fn` for native comparison. Sweep M=3,5,10 and N=12,50,200. Independent shooting-defect evaluation can be a second variant, but does different work from a rollout. |
| Gauss-Newton parameter fit | Fit `a*exp(-exp(k)*t)+b` to 16 observations, 3 unknowns, eight `fori_loop` steps, damped normal matrix, Cholesky and two triangular solves | Automatic differentiation, repeated positive-definite solves, exp, reductions and fixed iteration counts | JAX/XLA CPU; Scaly `roots.GaussNewton` or an explicit identical fixed-step loop. Sweep observations, parameter count and iterations. Match stopping rules before comparing built-in converging solvers. This smoke uses fixed eight steps. |
| Chain control QP | A condensed Gauss-Newton quadratic program from the chain rollout and Scaly's stage/terminal cost, with control bounds [-1,1]. M=3, N=4, 12 controls; a 15-step primal-dual `fori_loop` | Matrix assembly, fraction-to-boundary reductions, positive slacks and multipliers, repeated linear solves and substantial dispatch traffic | Exact same QP solved by Scaly's generated dense IPM here, plus JAX/XLA executing the identical 15-step algorithm. Scaly uses its own initialization, scaling, predictor-corrector and stopping policy, so it is a solver comparison, not an isolated compiler comparison. Sweep M, N, iterations, conditioning and warm starts. |
| Batched Newton | `jax.vmap(newton)` over four Bratu parameters, n=8 | Batched LAPACK, masked lane updates, batch-dependent convergence and throughput | JAX/XLA CPU and a Scaly `vmap` mirror. Sweep batch 1,4,16,64 and n. JAX's converging `vmap` runs until the slowest lane finishes; use fixed iterations for the first compiler comparison or report that extra work. |

The rollout and chain QP smoke cases use dt=0.002 and a straight-chain initial state. Scaly's canonical controller uses dt=0.2 and a suitable physical warm start. The QP comes from the actual chain equations and weights, but is one condensed subproblem, not the full nonlinear controller. Its construction occurs outside timing; Q, c, G and h are runtime inputs to the timed IPM function. H2 should use Scaly's canonical states and parameters and add outer SQP only after the QP path is verified.

### Compilation and correctness results

`check_workloads.py` calls both today's `aotax.compile(...).load()` and the temporary bridge compiler. The tests compare all output leaves against `jax.jit`, including iteration counts.

| Workload | Today's aotax | Bridge maximum absolute error | Runtime dispatch calls | Runtime foreign calls |
|---|---|---:|---:|---:|
| Chain scan | Pass | 1.1e-22 | 216 for 12 stages | 0 |
| Newton | IREE rejects `lapack_dgetrf_ffi` | 3.9e-16 | 174 for 3 iterations | 9 |
| Gauss-Newton fit | IREE rejects `lapack_dpotrf_ffi` | 1.7e-16 | 80 for 8 iterations | 24 |
| Chain QP IPM | IREE rejects `lapack_dgetrf_ffi` | 2.2e-16 | 1,336 for 15 iterations | 45 |
| Batched Newton | IREE rejects batched `lapack_dgetrf_ffi` | 4.5e-15 | 481 for 4 batch iterations | 12 batch calls |

The batched lane iteration counts are [3,3,3,4]. A batch foreign call loops over four matrices, including converged lanes. `workload_results.json` contains exact numerical errors and static dispatch-site counts. `workload_counts.json` contains measured runtime counts from an instrumented C host. Context sizes were 13,824, 24,064, 9,408, 77,248 and 61,248 bytes in table order, excluding the prototype adapters' stack scratch. Counts exclude context creation. They count dispatch invocations, not SIMD work or LAPACK calls inside a batch adapter. The IPM dispatch count includes JAX's pivot-to-permutation loops. Replacing its solve with Cholesky would reduce this work, but changes were deliberately kept separate from the initial mechanism.

## The custom-call mechanism

### Confirmed IREE route

The smallest successful input is:

```mlir
func.func private @foreign(tensor<4x4xf64>) -> tensor<4x4xf64>
func.func @main(%a: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %b = func.call @foreign(%a) : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %b : tensor<4x4xf64>
}
```

With the existing aotax compiler flags, IREE 3.12 emits:

```mlir
util.func private @foreign(%arg0: !hal.buffer_view) -> !hal.buffer_view
  attributes {iree.abi.stub}
// In main:
%0 = util.call @foreign(%arg0) : (!hal.buffer_view) -> !hal.buffer_view
```

The same happens in the real solve and in loop bodies. `native_main.c` links just the static archive, libm and Accelerate, runs without Python and verifies a pivoting solve against an all-ones solution. `routes.py`, `route-func.mlir` and `route-func-hal.mlir` are the minimal evidence. `solve4/iree/original.mlir`, `stablehlo.mlir`, `hal.mlir` and `host.c` show the whole route.

For each supported custom-call specialization, `bridge.py` creates a unique private tensor declaration and a `func.call` with the original tensor operands and result types. A Python registry retains the target, static shapes, dtypes and backend configuration under that declaration's symbol. IREE handles all surrounding arithmetic, buffer exports/imports and host loop lowering. The registry tells the C generator how much result storage to allocate and which adapter to call. Generated adapter wrappers should have internal C linkage so two compiled libraries can link into the same application.

The C adapter takes input pointers followed by output pointers and returns an integer execution status. The host supplies the result storage. The HAL call's SSA results become those pointers. The host does not construct IREE buffer-view objects or load an IREE module. Its existing representation already treats a buffer view as a pointer at the view's offset.

Keep the explicit tensor results until HAL. An attempted destination-buffer approach used `tensor.empty`, `hal.tensor.export`, a void external call and `hal.tensor.import`. It compiled, but IREE scheduled computations reading the supposed results before the external call. The IR did not express the mutation as tensor data flow. `unsafe_empty_export.py` and the initial `route-buffer*` artifacts record that rejected approach. It must not become the implementation. Proper destination-style `flow.call` is another possible route, but was not needed once the plain declaration worked. The attempted hand-written `flow.func` form had a syntax error, so this spike makes no claim that flow calls are unsupported.

### Host changes

The temporary host extension is about 25 added Python lines:

1. Recognize registered bodyless `util.func` declarations and exclude them from generated internal functions.
2. Lower `util.call` by allocating static result storage, assigning each result pointer and calling the adapter.
3. Use the existing control-flow liveness information and buffer-slot pool rule for results allocated in cycles.
4. Translate `hal.device.queue.copy` to an ordered CPU memory copy. The FFI loop naturally introduced this operation.

The prototype skips every external declaration and trusts its registry. Production must accept only registered symbols, validate their lowered argument/result signatures and reject unknown calls. Refactor the existing allocator's slot-selection code for reuse rather than maintaining two implementations. Expect about 50-80 changed host lines after validation and shared allocation logic. The rewrite, LAPACK adapters and user FFI description belong in a new module, not as a large addition to `host.py`. Splitting the program into compiled segments around custom calls would require a second implementation of loop carries, branches and buffer lifetimes. It was not prototyped because the external-call route preserves those through IREE already.

A reasonable first production implementation is roughly 300-500 additional lines across those concerns, plus focused correctness tests. This is an estimate, not an implementation commitment.

### LAPACK contracts

The current JAX export is the contract to implement, rather than a guessed call to `dgesv`:

- `jnp.linalg.solve` emits one `lapack_dgetrf_ffi` and two `lapack_dtrsm_ffi` calls in JAX 0.11.2. JAX converts the one-based LU pivots to a permutation, gathers the right-hand side, then performs the two triangular solves. `getrs` is not emitted by this export.
- `jnp.linalg.cholesky` emits `lapack_dpotrf_ffi`, followed by JAX's status selection and triangular masking.
- `jax.scipy.linalg.solve_triangular` emits `lapack_dtrsm_ffi`. Parse and validate `side`, `uplo`, `trans_x` and `diag`. The prototype handles their real-valued enum cases with row-major CBLAS.
- Implement s/d getrf, potrf and trsm first for the existing float32/float64 public types. Add getrs or potrs when a recognized export or an explicit solver adapter needs them. Their adapters need pivots, transpose/uplo, matrix dimensions and right-hand-side count. Do not require a JAX whole-solve rewrite to enable the first workloads.

JAX describes LAPACK operands in column-major layouts. IREE's tensor import/export ABI gives this CPU host dense row-major arrays. The adapter must convert the logical matrix. The prototype copies A into a column-major temporary, factors it, and copies the factor back into the row-major result. It uses the row-major C interface to the Basic Linear Algebra Subprograms (CBLAS), through `cblas_dtrsm`, for the solves. This is correct for nonsymmetric matrices and pivoting; factoring the row-major bytes as if they were A would factor its transpose and be wrong.

An optional `dgetrf_` + `dgetrs_` adapter is implemented by `paired_lapack.py`, invoked through `jax.ffi.ffi_call`. It demonstrates a direct fused solver target, but does not replace ordinary `jnp.linalg.solve` automatically. A later optimization may recognize the entire LU/permutation/triangular-solve sequence or expose a dedicated solver FFI function. Preserve arbitrary LU consumers and differentiation-generated uses when considering such a rewrite.

### Ownership, copies, loops and batch dimensions

Inputs are borrowed and immutable. Every custom-call result initially receives fresh context-owned storage. LAPACK mutates only its temporary and output buffers. JAX's custom-call alias annotations are not permission to modify the public caller's array. A live input, constant or carried iterate must survive a factorization.

Inside a host cycle, a fixed result slot can overwrite a value carried from a previous iteration. Allocate a pool using the existing live-buffer-count-plus-one policy and select a slot disjoint from the live pointers. The prototype applies this policy separately to every foreign result. Normal HAL uses already participate in liveness; synchronous C calls need no lifetime extension beyond the call. Results that reach the public output follow the existing final-copy logic. Direct public output binding can be optimized later.

The prototype's transpose scratch and pivots use bounded C stack arrays, at most 32 KiB of matrix scratch at n=64. Production should put scratch in the context, allocate it once and reuse it across synchronous calls. Neither the foreign adapter nor the generated host should allocate heap memory per evaluation. Keep scratch separate from live returned factors. Dense adapters should loop over flattened static leading batch dimensions and write a separate info scalar and pivot segment per matrix. JAX's own surrounding broadcasts handle ordinary broadcasted batch operands; H1 must test those cases explicitly rather than assume matching batch strides.

`check_bridge.py` verifies nonsymmetric pivoting, two right-hand sides, batches of three matrices, preservation of both caller input and an explicitly returned original matrix, Cholesky, invalid Cholesky and a singular solve. It calls the same contexts repeatedly with changed inputs. The prototype is limited to square float64 matrices, tested batch layouts and nonempty dimensions. Rectangular LU, float32, empty sizes, all triangular options, broadcasted batches and overlap/alignment combinations remain production tests and adapter work.

### Numerical status and execution failure

LAPACK `info` is a tensor result, not a host execution error. Keep it in the graph so JAX's existing selection produces its established NaN behavior. The tested singular solve returns all NaNs on both paths; an indefinite Cholesky produces the same NaN mask. LU's returned pivots remain one-based because JAX performs its own subtraction. Do not subtract in both places.

A nonzero C adapter return reports execution failure and follows the existing `{name}_call` error path, including floating-point-mode restoration. A user's solver should return convergence status, iterations and residual as ordinary output buffers. Reserve adapter failure for inability to execute, not failure to converge. Dimension, dtype and unsupported-attribute errors should fail compilation before LAPACK sees them. Preserve the existing flush-to-zero policy and test numerical edge cases under it.

## User FFI and linking

`link_ffi.py` exports an actual `jax.ffi.ffi_call('h0_scale', ...)` inside a five-step JAX loop. It replaces the generated adapter's body with a call to `user_scale`, compiles `ffi_impl.c` separately and merges its object into the archive. Loading the library returns exactly `32*x`. The example implementation is intentionally small; it proves target resolution, loop placement and link-time implementation separately from LAPACK.

Production should distinguish a JAX target name from its C symbol. A target registered with `jax.ffi.register_ffi_target` generally has an XLA FFI call-frame ABI. It is not a function taking raw array pointers. Do not call that pointer using a guessed signature or require the XLA runtime in a deployment library. Require an explicit aotax C adapter unless a user deliberately supplies an XLA-frame compatibility adapter.

A narrow initial interface can register target name, C symbol, supported static signature/attributes and scratch bytes. The generated call header should describe inputs and results. For reusable adapters across shapes, a small common C descriptor should carry each buffer's data pointer, dtype, rank and static shape. Pass typed scalar backend attributes or a documented immutable attribute encoding; never silently discard `mhlo.backend_config`. Allocate optional scratch in the context. Avoid solver-specific lifecycle callbacks until a real target requires them.

The application may link a user object or library after `libmodel.a`:

```sh
cc main.c libmodel.a my_solver.o -lm -framework Accelerate
```

The static archive can intentionally retain unresolved user symbols. The generated header lists required libraries. Python's `Library.load` needs an explicit extra-link-arguments option so the same artifact can be tested. Bundling user objects, as the prototype does, is another valid choice, but should be explicit. Linking and loading an existing solver still needs its headers, dependent libraries and compatible integer widths. Scaly generated pure solver C only needs its generated header/source and workspace; IPOPT, PIQP and acados wrappers add their own dependencies and state.

JAX export rejects unregistered custom targets unless supplied with `DisabledSafetyCheck.custom_call(target)`. The bridge supplies this exemption for its known prototype targets. Production must derive exemptions only from the explicit target registry. Aotax does not need to register an XLA handler merely to export and compile a call, but an XLA comparison of that FFI workload does need an XLA-compatible registration.

Start user FFI with synchronous CPU calls and static dense tensor inputs/results. Reject unsupported tokens, ordered effects, opaque state, layouts and alias requests with a clear compilation error. LAPACK adapters deliberately implement fresh-output semantics. General user input/output aliasing needs an explicit adapter contract and copy-in behavior; it must not inherit LAPACK's assumptions. A foreign library retaining borrowed pointers after return is outside this first interface.

## Measurements

The main measurements interleave native aotax, Python-loaded aotax and warm `jax.jit` calls in five fresh processes. Each process rotates measurement order and takes the best of three batches; reported values are the best of five processes. Compilation and context creation are excluded. Native timings call the generated C entry point in a C loop with preallocated input and output arrays. Python-loaded aotax includes its NumPy validation, allocation and ctypes overhead. JAX uses already resident CPU arrays and blocks for every result.

JAX timings include Python/XLA invocation cost; native C timings do not. These are deployment latency comparisons, not proof that the native numerical kernels beat XLA by the same ratio. Python-loaded aotax is included to make that boundary visible. The 20-solve function puts the loop inside both compiled programs, changes A each iteration to prevent factor hoisting and amortizes invocation cost. The machine was shared; do not treat sub-microsecond differences as stable rankings.

### Unchanged `jnp.linalg.solve`

All entries are microseconds per call. The last two columns time a whole 20-iteration `fori_loop`.

| n | Native aotax | Python-loaded aotax | JAX/XLA CPU | Native loop of 20 | JAX loop of 20 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.305 | 3.215 | 3.661 | 6.041 | 11.389 |
| 8 | 0.569 | 3.478 | 3.844 | 11.527 | 14.376 |
| 16 | 1.260 | 4.201 | 4.555 | 27.081 | 37.939 |
| 32 | 4.680 | 8.617 | 15.406 | 104.795 | 92.075 |
| 64 | 14.808 | 20.126 | 30.313 | 328.594 | 335.395 |

Native execution wins on the small complete programs, but the n=32 loop loses to XLA and n=64 is close. H1 enables useful control flow without establishing a general numerical-kernel speedup.

### Pure JAX factorization alternative

`bench_solve.py` also calls JAX's private `_lu_python` explicitly and implements forward/back substitution with JAX loops. For n<=64, the factorization lowers to ordinary StableHLO and needs no LAPACK custom call. It compiles with the bridge compiler without any custom calls and matches the finite, conditioned test cases.

| n | Native pure-JAX LU solve | Native LAPACK bridge |
|---:|---:|---:|
| 4 | 0.665 | 0.305 |
| 8 | 1.722 | 0.569 |
| 16 | 5.383 | 1.260 |
| 32 | 32.101 | 4.680 |
| 64 | 219.893 | 14.808 |

The private JAX implementation is useful experimental evidence, not a production dependency. Its block size is 128 and larger blocked cases call triangular solves again. `check_pure_today.py` also confirms compilation and correct loading through the completely unchanged aotax path at n=4 and n=64. Generic `lax.linalg` CPU operations still select the LAPACK lowering. JAX's generic Cholesky and triangular-solve lowering emit StableHLO operations, but switching global private lowering registrations would be brittle and affect other exports. A concrete alternative-backend export was also tested in `generic_lowerings.py`. TPU-platform export emits `stablehlo.cholesky` and `stablehlo.triangular_solve`; IREE 3.12 rejects both. Its LU is another rejected custom call, `LuDecomposition`. Choosing another JAX export platform therefore does not remove this blocker. A maintained pure-JAX factorization is a possible portability option later. It is a poor first choice for these CPU workloads and does not solve general FFI.

`paired_lapack.py` measured an explicit FFI solver using `dgetrf_` plus `dgetrs_` at 0.294, 0.585, 1.060, 3.061 and 9.625 microseconds for n=4,8,16,32,64. These were a single process, different matrices and best of five batches, so they are directional only. The reduction at larger n suggests profiling the JAX pivot conversion and copies before optimizing LAPACK itself. This fused function is not the unchanged-JAX solve used in the main table.

### Complete smoke workloads

Microseconds per entire invocation, from `workload_timings.json`:

| Workload | Native aotax | Python-loaded aotax | JAX/XLA CPU |
|---|---:|---:|---:|
| Chain scan, M=3, N=12 | 1.587 | 5.729 | 8.963 |
| Newton, n=8, 3 iterations | 2.083 | 6.226 | 7.087 |
| Gauss-Newton fit, 8 steps | 1.180 | 4.769 | 6.560 |
| Chain QP, 12 controls, 15 steps | 19.763 | 24.502 | 43.400 |
| Batched Newton, batch=4, n=8 | 8.423 | 12.570 | 18.655 |

Two Scaly comparisons were built and run. `scaly_baseline.py` constructs the exact Bratu residual and `roots.Newton` with dense LU and the same termination rule. Both solvers take three iterations. Scaly returns a scalar residual and a status while the JAX smoke returns the residual vector; the core solve agrees to 2.8e-17. `scaly_qp.py` solves the same condensed chain QP using the built-in generated dense IPM. It takes five iterations, versus fifteen in the simpler JAX algorithm, and agrees in the primal solution to 8.5e-9. Its output includes extra status/dual information.

Native Scaly compilation uses `-O2 -mcpu=native -fno-math-errno`. These generated solvers require no external solver library and reported zero caller workspace. `bench_scaly.py` measures both native libraries and XLA in each of five fresh processes:

| Problem | Scaly generated C | Native aotax | JAX/XLA CPU |
|---|---:|---:|---:|
| Newton | 0.515 | 1.969 | 6.920 |
| Chain QP | 2.867 | 19.699 | 46.451 |

The Newton comparison exposes remaining compiler and dispatch cost. The QP comparison also includes different algorithms and iteration counts, so the 6.9x gap must not be attributed entirely to aotax. The direct source import has no installed Scaly plugin metadata; `scaly_qp.py` imports `IPM` from `scaly.opt.ipm.method` explicitly rather than relying on `sc.opt.IPM` discovery.

## H1 and H2 plan

### H1a: preserve foreign tensor calls through HAL

Add a custom-call rewrite module and a compile-time registry for target specializations. Implement private tensor declarations and calls, registered external declaration handling, synchronous C calls, result allocation through shared liveness-aware slot selection and HAL queue copies. Keep the existing public call ABI and runtime-free archive. Unknown targets must remain clear compilation errors.

Acceptance is an ordinary C program linked to the archive that executes an unchanged `jnp.linalg.solve` inside a loop, preserves its inputs and matches JAX. Include a loop carrying old and new results to test output lifetime, as well as shifted caller overlap and unaligned pointers. Preserve strict rejection of unsupported HAL patterns. No harness change is needed for this phase.

### H1b: numerical adapters

Implement float32 and float64 LU, Cholesky and triangular-solve adapters with static shape/configuration validation, batch handling and context-owned transpose scratch. Cover matrix right-hand sides, rectangular LU if supported, transpose/side/unit diagonal combinations, broadcasting, singularity, non-positive-definite inputs and zero dimensions. Keep info and pivot conventions identical to JAX. Add getrs/potrs only for actual target forms or explicit fused solver adapters.

Run focused new checks, then the existing pytest, Ruff, format, type checks and existing benchmark correctness checks. Follow the project's cross-family review procedure once the implementation passes. Do not optimize away the pivot-conversion graph in the first correctness milestone.

### H1c: user C targets

Expose the registry through a small explicit compile option and publish a C adapter convention. Add selective JAX export exemptions and Python extra link arguments. Demonstrate a separate solver object linked after the aotax archive and called inside a loop. Keep numerical convergence status in tensor outputs. Test missing symbols, unsupported attributes and rejected effects at the boundary. A generated Scaly pure solver makes a useful first realistic foreign target because its workspace and C ABI are available without another solver installation.

H1a and H1b are sequential. LAPACK adapter implementation can start alongside host support once the adapter contract is fixed. User-facing FFI should follow the proven internal contract. Stateful IPOPT/acados solver instances and callback-driven oracle integration need a later design; target-name mapping alone does not provide their lifecycle or callbacks.

### H2: benchmark useful complete programs

The harness generalization proceeds independently. Coordinate only the workload-module contract, native reference hooks, multi-output layout and integer result validation. Do not edit the other agent's harness during H0.

1. Add chain rollout first. It compiles today and establishes control-flow correctness and a JAX timing column before H1 lands. Add the exact Scaly scan mirror.
2. After H1b, add Newton, the Cholesky least-squares loop and batched Newton, using the prototype definitions as small smoke cases. Verify solutions, residuals, status and iteration counts before timing. Separate fixed-work and convergence-dependent rows.
3. Add the condensed chain QP. Compare identical 15-step JAX programs through aotax and XLA. Report Scaly IPM separately as a complete-solver baseline, with its iterations and residuals. If a compiler-only comparison against Scaly is needed, port the same algorithm and initialization on both sides.
4. Use canonical Scaly chain inputs, then a full SQP outer loop and larger npmpc/race-car problems. Preserve sparsity or stage structure before claiming scalability. A dense condensed solve scales cubically with horizon and will not fairly compete with Scaly's stagewise KKT factorization.
5. Record runtime dispatch count, foreign-call count, context size, iteration count and factor reuse alongside latency. Profile the worst complete-workload ratios. Only then consider fusing LU/permutation/solve, stagewise linear algebra, dispatch fusion or a sparse solver FFI target.

H2's rollout and reference work can run before H1. Linear-solve rows depend on H1b. Real FFI benchmarks depend on H1c. The full nonlinear-controller milestone depends on verified QP behavior, canonical inputs and explicit algorithm parity choices.

## Reproduction

Run from the aotax worktree so `uv run` uses its environment. All outputs remain in `/tmp/aotax-h0`.

```sh
uv run python /tmp/aotax-h0/inspect_h0.py
uv run python /tmp/aotax-h0/routes.py
uv run python /tmp/aotax-h0/generic_lowerings.py
uv run python /tmp/aotax-h0/check_pure_today.py
uv run python /tmp/aotax-h0/bridge.py
cc /tmp/aotax-h0/native_main.c -I /tmp/aotax-h0/solve4 /tmp/aotax-h0/solve4/libsolve4.a -framework Accelerate -lm -o /tmp/aotax-h0/native_main
/tmp/aotax-h0/native_main
uv run python /tmp/aotax-h0/check_bridge.py
uv run python /tmp/aotax-h0/check_workloads.py
uv run python /tmp/aotax-h0/profile_workloads.py
uv run python /tmp/aotax-h0/link_ffi.py
uv run python /tmp/aotax-h0/paired_lapack.py
uv run python /tmp/aotax-h0/bench_solve.py --prepare
uv run python /tmp/aotax-h0/bench_solve.py --reps 5
uv run python /tmp/aotax-h0/bench_workloads.py --prepare
uv run python /tmp/aotax-h0/bench_workloads.py --reps 5
uv run python /tmp/aotax-h0/extract_scaly.py
uv run --with scipy python /tmp/aotax-h0/scaly_baseline.py
uv run --with scipy python /tmp/aotax-h0/scaly_qp.py
uv run --with scipy python /tmp/aotax-h0/bench_scaly.py
```

`check_workloads.py` deliberately invokes failing original compilations and catches them, so its log contains expected IREE errors. It records each result rather than failing the entire run. `check_bridge.py` raises on correctness failure. The `host_prototype.py` file is a generated copy of the original host with the temporary additions; it is imported dynamically by `bridge.py` through its own namespace, never installed over aotax. `timing.py` contains the native timing helper. JSON files retain raw per-process timing data, summary timings, correctness results and dynamic call counts. Generated archives, headers, C and intermediate representations live in their named subdirectories.
