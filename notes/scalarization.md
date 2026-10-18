# Selective scalarization in aotax

Research spike for PLAN.md, phase 3 item K, on main at ce1fc4d. Its prototypes, generated libraries, native timing binaries and raw measurements were written to `/tmp/aotax-k`, outside the repository, and are not kept; the paths below refer to that scratch directory.

## Decision

Implement the power-of-two small-dimension tile rule first, then consider a bounded C path for wholly tiny dispatches if small-program latency is a priority. In the focused confirmation, these give 15.7% lower latency on race_cars N=500 and 25.6% lower latency on race_cars N=5 respectively. They do not close the Scaly gap on chain. Do not globally enable every scalarization idea: both large-loop C generation and whole-program single-dispatch grouping lose, and disabling multi-use fusion regresses N=5 even though it helps chain_jac.

The confirmed general tile rule changes only 2/4/65/65/9 generic configurations on the five workloads. It fills eligible power-of-two extents at most eight, keeps the native vector register width at 16 bytes, caps the virtual vector tiling level at 32 elements and leaves other dimensions to IREE. No workload or shape names enter the rule. The narrow permutation-only rule failed to recover the full benefit.

The broad sweep and the focused confirmation are separate interleaved runs. Their absolute latencies differ with machine load; compare each variant with the baseline in its own table. Tiny effects should not be treated as wins.

Scaly's advantage comes from a preserved local function boundary, scalar substitution across that whole function, common subexpression sharing, and a stage loop outside it. Merely replacing each existing dispatch with scalar C does not recreate that organization. The derivative work in item J remains necessary for the large gap.

## Snapshot and method

- aotax `ce1fc4da61dd20227770e579ebd99a480a187e67`, the supplied worktree's main snapshot.
- Scaly `edb3cbb3f545bf661841d7df3f2b46fb2b94d2fc`. The benchmark dependency and `/Users/tudoroancea/dev/scaly` match this revision.
- JAX 0.11.2, asdex 0.5.2, iree-base-compiler 3.12.0. The compiler identifies itself as `3.12.0rc20260917`, commit `2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d`, LLVM 24.
- Apple M4 Max, 48 GiB, Apple clang 21.0.0. Float64, CPU, one aotax context per process, unchanged aotax math preparation and Accelerate routing.
- `study.py` obtains the exact existing workload function from `Workload.aotax`. Each result element is checked against `jax.jit` of that function. Each native process checks all stored output values before timing. `verify_native.py` additionally checks four deterministic 1% input perturbations per variant. This checks compiler equivalence to JAX, not the separate correctness of asdex's sparsity detection.
- Inputs and outputs in native timing are 64-byte aligned. Calls run in a calibrated native C loop for at least 0.15 seconds of process CPU time. Context creation, loading and checks are excluded. Every measurement starts a fresh process; variants are shuffled and interleaved across five repetitions. Reported times are the minimum of those five processes. Raw repetitions remain available. The machine is shared; differences below about 5% are inconclusive.
- Scaly uses its own sparse coordinate order and its own appropriately reordered reference. It is compiled with the benchmark's `-O3 -mcpu=native -fno-math-errno`, `lanes=auto`, scalar libm. aotax keeps its normal IREE flags and host `-O2`, except the explicitly labeled host/C-codegen experiments.
- Instrumented per-dispatch times use `mach_absolute_time` and are for attribution. They include timer cost, use only one profiling run, and must not be read as precise end-to-end comparisons. Matrix products routed to Accelerate are not all included in dispatch timing totals.

Sources read: AGENTS.md, PLAN.md F2 and K, README.md, benchmarks/README.md, benchmarks/harness.py, workloads/chain.py, race_cars.py, npmpc.py and their `_chain`, `_race_cars`, `_npmpc` implementations, src/aotax/__init__.py, host.py, blas.py and math.py. Scaly sources are listed below. Existing F2 logs supplied context; measurements and scripts here were written independently.

## What Scaly actually does

There is no universal rule such as "an array with at most ten elements becomes scalar."

1. Expression lowering first produces buffers, stores, pure calls, and statically bounded loops. `passes/lowering/elementwise.py` starts with one flattened loop per elementwise operation. `passes/lowering/calls.py` lowers a vectorized map, or VMAP, to an outer loop that calls a local function with offset input and output pointers. Broadcast arguments have stride zero.
2. `passes/lowering/ctx.py` disables automatic scalarization of the top-level entry. Pure callees can be scalarized automatically. A `.scalar()` hint overrides automatic limits, including at the entry; `.block()` or an opaque hint wins over scalar hints. All values must be float64 because substitution must not erase narrowing stores. `function/concrete.py` computes this policy; derived functions inherit it. The chain equality-stage definition explicitly requests `.scalar()`.
3. `passes/program/scalarize.py` interprets eligible procedures symbolically. It expands constant-bound positive-step loops and eligible pure calls, replaces temporary buffer elements with expression values, and constant-folds arithmetic. Buffer copies, views and transposes become indexing in those expressions. It does not scalarize effects or unknown/dynamic control flow.
4. Automatic expansion requires estimated parameter/buffer/store expansion work at most 65,536 per procedure, at most 4,096 distinct scalar operations in the expanded procedure, and at most 16,384 operations plus statements of accumulated program growth. An explicit scalar hint bypasses those automatic size limits. Hoisted prologues can be eligible for expansion only with their caller.
5. `scheduling.py` shares common scalar nodes and introduces named values for repeated loads/expressions or expression depth of 32. Thus a local stage becomes straight-line assignments with common subexpressions shared before output stores. This pass does more than asking the C compiler to unroll loops.
6. `hoist_invariant.py` first splits out work depending only on broadcast map arguments. Afterwards, `fold_tiles`, range fusion, elementwise producer fusion, arithmetic folding, unit-loop unrolling, workspace packing and store coalescing remove further intermediates. Elementwise fusion only moves a single-consumer private producer when it will not increase its execution count. It refuses repeated expensive transcendental expressions and unsafe movement across writes.
7. `widen_ranges.py` then widens suitable independent mapped/contiguous loops or supported ordered reductions. Mapped callees are inlined into the loop. The lane choices are 1, 2, 4 or 8, limited by loop length and estimated peak live floating values. A width fits when `peak_live * width <= 4 * register_slots`. On this build, ARM chooses two nominal lanes and 64 register slots; a large local stage can lower its cap to one. Explicit vector helpers retain scalar per-lane libm calls on macOS. It does not vectorize a tiny state axis merely because it exists.
8. Large products retain loops. `lowering/contraction.py` arranges the reduction loop outside independent output loops for contiguous vectorization where useful, and handles matrix-vector products with four explicitly interleaved output-row accumulators.

Generated evidence, from `scaly_summary.py` and each `scaly/generated.c`:

- chain_jac M=3 has 384 scalar assignments in the optimized entry, 10 remaining loop nodes and 4,320 doubles of workspace. The surviving entry's `scalarized=False` does not mean its local callees were never expanded; later inlining and pruning have removed them.
- chain M=3 has 1,269 assignments in the entry, 168 loop nodes, a small scalar hoisted helper and 26,028 doubles of workspace. Scaly also keeps intermediates where full expansion is too costly.
- race_cars N=5 and N=500 generate the same 1,188-line C structure, with 772 assignments in the entry, five loop nodes, three mapped loops and zero workspace doubles. Only outer trip counts change. This is the clearest example of selective stage scalarization.
- npmpc N=12 has 2,261 source lines, retained product/reduction loops, and zero externally allocated workspace doubles. Zero workspace does not imply no local storage or no loop buffers.

## What IREE produces

| Function | Unique | Calls | Main executables with LLVM loops | Logical dispatch stores/call |
|---|---:|---:|---:|---:|
| chain_jac M=3 | 37 | 83 | 35/37 | 1.673 MB |
| chain M=3 | 82 | 180 | 76/81 | 2.243 MB |
| race_cars N=5 | 124 | 229 | 58/123 | 0.046 MB |
| race_cars N=500 | 124 | 229 | 118/123 | 4.521 MB |
| npmpc N=12 | 77 | 76 | 50/76 | 0.187 MB |


"Unique" counts dumped executable definitions including initialization helpers. "Calls" counts dispatch invocations during one kernel evaluation, so repeated fourth-order Runge-Kutta bodies reuse executable definitions. IREE can additionally route zero fills through a builtin executable. Current counts differ from the older F2 count quoted in the brief; the exact current HAL and counting scripts are retained.

- chain_jac M=3 has explicit logical reductions over 3, color batches of 14, stages/mass links of 40/80, and flattened elementwise loops of 3,360, 5,040, 6,720 and 6,846. Only two invocations with generic bodies have all logical axes at most 16. The common matvec-like dispatch has loops 14×80×3; it runs three times and accounts for about 8.9 µs in the instrumented profile.
- chain M=3 adds a 12-color Hessian axis. Common shapes are 12×80×3 and 12×360. There are 23 distinct generic axes of extent 3 and 29 of extent 12 in the dumped definitions. Only five generic invocations have all axes at most 16. The small dimensions are usually nested within large color/stage arrays.
- race_cars N=5 has 53 generic axes of extent 4, 76 of extent 5 and 19 of extent 6 across unique definitions. 151 generic invocations have all axes at most 16. Every dumped definition's largest tensor is at most 256 elements; the direct-C implementation supports all main executable definitions, leaving the builtin fill helper to IREE.
- race_cars N=500 preserves those small color axes but replaces the horizon axes by 500 and 501. Common layouts alternate 500×4, 4×500 and 4×4×500. Only five generic invocations have all axes at most 16. The two calls to `main_dispatch_168_elementwise_transpose_4x4x500_f64` take about 22.4 µs together; dispatch 145 of the same shape adds about 11.1 µs.
- npmpc N=12 contains 12/32-wide product loops, a contraction dimension 9 and color/state axes 6, 4, 5, 2 and 3. 27 generic invocations have all logical axes at most 16, but these account for only about 1 µs of instrumented dispatch time. The large neural-network products dominate; normal Accelerate routing remains valuable.


These are often batched small axes, not wholly small tensors. After asdex, a chain state of nine entries has become arrays with a 12- or 14-color axis, a stage/mass axis of 40 or 80, and coordinates of size three. In race_cars, the four-entry color axes survive alongside the horizon axis. Flattening or unrolling one four-entry axis cannot remove the large color arrays or their zero-filled regions.

Measured attribution:

| Function | Host fills, calls/bytes/µs | Host copies, bytes/µs | Dispatches named transpose, µs |
|---|---|---|---:|
| chain_jac M=3 | 27 / 1,022,448 / 9.03 | 15,392 / 0.19 | 0.00 |
| chain M=3 | 52 / 1,084,512 / 9.71 | 156,512 / 2.77 | 0.00 |
| race_cars N=5 | 63 / 33,136 / 0.35 | 816 / 0.03 | 0.39 |
| race_cars N=500 | 63 / 3,161,536 / 18.23 | 72,096 / 1.14 | 56.18 |
| npmpc N=12 | 18 / 43,344 / 0.49 | 14,552 / 0.29 | 0.69 |


The `dispatch_store_bytes` figure sums logical tensor bytes stored by each executable invocation. It is not measured memory traffic and excludes internal loads and scratch, host copies, host fills and any eliminated stores. It nevertheless shows the scale of values crossing dispatch boundaries. `profile/profile.txt` lists every call site; `profile.json` aggregates repeated executable names.

Each case also has `configs/configured.mlir`, `llvm/*.optimized.ll`, `.codegen.ll`, assembly, and `llvm-loops.json`. The source/configuration files give exact logical loop sizes; the optimized LLVM files verify which loops survive code generation. For example, N=5 already removes many loops, yet 58 of 123 main executables still contain a control-flow cycle. N=500 has cycles in 118 of 123. Therefore "IREE never unrolls tiny loops" is false, while "IREE preserves the entire tiny program as scalar values" is also false.

## Mechanisms prototyped and measured

Each cell gives microseconds followed by candidate/baseline. Lower is better. "Not run" means precisely that. No performance claims come from failed compilations or fallback-only prototypes.

| Mechanism | chain_jac M=3 | chain M=3 | race N=5 | race N=500 | npmpc N=12 |
|---|---:|---:|---:|---:|---:|
| Baseline, µs | 65.050 | 85.946 | 3.598 | 261.154 | 16.269 |
| Scaly, µs | 4.001 | 23.658 | 0.686 | 50.337 | 15.315 |
| Multi-use fusion on | 65.895 (1.01×) | 87.540 (1.02×) | 3.595 (1.00×) | 259.392 (0.99×) | 16.252 (1.00×) |
| Multi-use fusion off | 57.377 (0.88×) | 81.414 (0.95×) | 4.235 (1.18×) | 248.274 (0.95×) | 16.537 (1.02×) |
| Elementwise multi-reduction fusion | 65.026 (1.00×) | 87.479 (1.02×) | 3.595 (1.00×) | 261.422 (1.00×) | 16.259 (1.00×) |
| Propagate collapse across expands | 65.016 (1.00×) | 86.073 (1.00×) | 3.608 (1.00×) | 260.784 (1.00×) | 16.293 (1.00×) |
| LLVM bounded unrolling | 71.739 (1.10×) | 81.997 (0.95×) | 3.541 (0.98×) | 260.565 (1.00×) | 16.944 (1.04×) |
| Native vector width 8 bytes | 75.967 (1.17×) | 102.496 (1.19×) | 4.134 (1.15×) | 308.432 (1.18×) | 18.579 (1.14×) |
| LLVM loop and SLP vectorization off | 66.047 (1.02×) | 87.276 (1.02×) | 3.619 (1.01×) | 260.999 (1.00×) | 15.431 (0.95×) |
| Full tiles for small dimensions | 68.491 (1.05×) | 81.894 (0.95×) | 3.569 (0.99×) | 216.551 (0.83×) | 16.798 (1.03×) |
| Unit vector tiles | 74.121 (1.14×) | not run | not run | not run | not run |
| Small tiles for permutation operations | 64.388 (0.99×) | 87.165 (1.01×) | 3.589 (1.00×) | 251.986 (0.96×) | 16.966 (1.04×) |
| StableHLO short sum expansion | 70.066 (1.08×) | 91.414 (1.06×) | 3.601 (1.00×) | 260.896 (1.00×) | 16.248 (1.00×) |
| StableHLO short dot expansion | not run | not run | not run | not run | 16.181 (0.99×) |
| Short sums and dots | not run | not run | not run | not run | 16.172 (0.99×) |
| Host C at O3/native | 65.265 (1.00×) | 86.237 (1.00×) | 3.636 (1.01×) | 256.832 (0.98×) | 16.225 (1.00×) |
| C dispatches, every tensor ≤256 | 65.595 (1.01×) | 89.904 (1.05×) | 2.669 (0.74×) | 258.812 (0.99×) | 16.150 (0.99×) |
| C dispatches, outputs ≤256 | 64.330 (0.99×) | 91.542 (1.07×) | 2.664 (0.74×) | 261.080 (1.00×) | 16.042 (0.99×) |
| C dispatches, outputs ≤64 | 65.411 (1.01×) | 87.267 (1.02×) | 2.594 (0.72×) | 256.252 (0.98×) | 16.087 (0.99×) |
| C dispatches with large outer loops | 69.861 (1.07×) | 111.589 (1.30×) | 2.685 (0.75×) | 391.174 (1.50×) | 17.017 (1.05×) |
| One dispatch, 1 MiB stack limit | not run | not run | 4.494 (1.25×) | not run | not run |

The focused confirmation includes the final power-of-two rule, with five new interleaved processes per cell.

| Confirmed variant | chain_jac M=3 | chain M=3 | race N=5 | race N=500 | npmpc N=12 |
|---|---:|---:|---:|---:|---:|
| base | 64.741 | 85.734 | 3.553 | 256.869 | 16.336 |
| scaly | 3.940 | 23.201 | 0.686 | 49.603 | 15.282 |
| power2small | 65.268 (1.01×) | 85.922 (1.00×) | 3.451 (0.97×) | 216.527 (0.84×) | 16.305 (1.00×) |
| fullsmall | 68.150 (1.05×) | 82.109 (0.96×) | 3.527 (0.99×) | 217.207 (0.85×) | 16.861 (1.03×) |
| fuseoff | 57.220 (0.88×) | 80.146 (0.93×) | 4.191 (1.18×) | 243.736 (0.95×) | 16.593 (1.02×) |
| strict256 | 65.403 (1.01×) | 86.653 (1.01×) | 2.645 (0.74×) | 257.831 (1.00×) | 16.232 (0.99×) |


Mechanism definitions:

- `multiuse` and `fuseoff` set `--iree-dispatch-creation-fuse-multi-use=true/false`. Disabling this can reduce duplicated producer work even though it increases dispatch count. This is separate from the aggressive-fusion experiments already rejected in AGENTS.md.
- `multireduce` sets `--iree-dispatch-creation-element-wise-fuse-multi-reduction=true`. `collapse` sets `--iree-dispatch-creation-propagate-collapse-across-expands=true`.
- `unroll` uses LLVM's `--unroll-threshold=4096 --unroll-full-max-count=16 --unroll-max-upperbound=16`. It changes backend profitability limits without enabling global IREE O3 or disabling distribution safeguards. Existing `--iree-llvmcpu-loop-unrolling` remains on.
- `vec8` tests an eight-byte native vector width, below the normal 16 bytes, as a scalar-code probe. The previously rejected 32/64-byte native widths were not repeated. `scalarllvm` disables LLVM loop and superword-level parallelism vectorization. IREE's higher-level vector IR can still remain, so these flags do not make a whole tensor graph scalar.
- `fullsmall` edits actual LLVM CPU lowering configurations after `--compile-to=executable-configurations`. It fills a vector tile dimension to its whole static extent when that extent is at most eight, while bounding each vector tiling level to 32 elements and preserving the 16-byte hardware vector width. It resumes with `--compile-from=executable-configurations`. Existing dispatch grouping and buffer layout stay fixed. This is a working lowering-configuration prototype, not an untested suggestion to use the Transform dialect.
- `unitvec`, tested on chain_jac, sets all nonzero vector tiling entries to one. `transpose_small` restricts the full-small rule to parallel generic operations whose operand maps actually permute full axes relative to the output map. It demonstrates that a dispatch's "transpose" name alone is not a useful tuning policy.
- `power2small`, if listed below, uses the same full-small rule only on power-of-two extents. It excludes awkward vector tiles of size three or six using a hardware-based rule rather than workload names.
- `pre_reduce` rewrites Stable High Level Operations sum reductions of at most eight elements as static slices plus elementwise adds. The pass handles an arbitrary batch shape and arbitrary reduction dimensions. It rewrites five reductions in chain_jac and ten in chain. Race has no qualifying reductions, so its rows are controls.
- `pre_dot` rewrites dot_general operations whose contracting-dimension product is at most eight as slices, transposes, broadcasts, products and ordered sums. Nine operations qualify in npmpc. Dispatch sites fall from 81 to 75, while unique executable definitions fall from 77 to 73. Small contraction removal is thus feasible, but fewer dispatches does not guarantee faster execution. `pre_both` applies both rules; npmpc has no qualifying sums, so it is the same transformation as `pre_dot`.
- `host_o3` recompiles the unchanged generated host with `-O3 -mcpu=native -fno-math-errno`. It controls for the C prototype's compiler flags.
- `strict256` emits C only when every tensor operand/result in a dumped executable has at most 256 elements. The selection uses executable types and operation support, never benchmark names. `tiny` bounds only the output size to 256; `tiny64` bounds output size to 64. These separate whole-tensor eligibility from the looser possibility of small outputs reading larger input views.
- `c_loops` allows large outputs. It retains outer loops and expands a suffix of small axes, with a 64-element expansion budget and short reductions. It is the direct test of "just scalarize the little dimensions in each existing dispatch."
- `one_bigstack` manually groups the entire pure N=5 tensor program in one `flow.dispatch.region`, with a single workgroup. The default 32 KiB limit fails with a 74,176-byte stack requirement. Raising the limit to 1 MiB lets it compile and pass the same reference checks, but it is slower. The source still contains 117 generic operations, 63 tensor pads, 59 extraction slices and 36 insertion slices. Grouping alone retains many temporary arrays.

The first version of the whole-program grouping script left the original graph live. That invalid experiment is not measured here. The corrected script explicitly rewires the export. `one_dispatch/bench.stale` is retained only to prevent it being mistaken for a valid single-dispatch result.

A further `whole_c` probe attempted to feed that grouped program into the bounded C translator. It rejected dynamic pad indexing and replaced zero sites. Its fallback binary was renamed `bench.fallback` and is excluded from timings. Supporting arbitrary dynamic pad/insertion indexing without speculative out-of-bounds loads needs additional code. This is a prototype limitation, not evidence that a whole-program scalarizer is impossible.

### Why ordinary host-hoisting options do not solve this

The pinned compiler's [HoistUniformScalarCompute.cpp](https://github.com/iree-org/iree/blob/2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d/compiler/src/iree/compiler/DispatchCreation/HoistUniformScalarCompute.cpp) only hoists pure arithmetic on integer/index scalars and one-element integer/index tensor extracts whose dependencies are already outside the dispatch. Float64 scalar dynamics remain compute dispatches. Its local copy is `HoistUniformScalarCompute.cpp`.

No documented small-tensor option in this compiler's complete `--help-hidden` output turns an arbitrary float tensor program into one C-host computation. The explicit one-region prototype tests the closest grouping alternative. The existing inline-static execution mode and previously rejected aggressive-fusion/global-O3 experiments were not repeated.

IREE's [tuning documentation](https://iree.dev/reference/tuning/) describes Transform-dialect specifications as a way to attach compilation/lowering settings. The split compilation used here establishes the same concrete capability for the pinned CPU lowering configurations. A production implementation can attach those attributes through a small matching transform or a controlled split compilation. It must not generate shape-specific specs from benchmark names. Tuning acts inside a dispatch; it does not recreate stage boundaries or remove cross-dispatch storage by itself.

## C code generation for tiny dispatches

This option is practical and has a measured use, but it should remain bounded.

`tiny_codegen.py` parses the same executable source files that aotax already reads for BLAS and binding-write analysis. It represents tensor loads and views as lazy index expressions, scalarizes short sum reductions, names common scalar expressions, then emits plain C next to the existing generated host. The host calls these helpers directly instead of constructing an executable ABI dispatch. Original IREE dispatches remain the fallback. The prototype reuses the original host's pointer offsets, alias/liveness analysis and floating-point-mode management.

The useful initial subset is:

- static float64 tensors plus integer/index address calculations;
- `hal.interface.constant.load`, binding subspans, static dispatch tensor loads/stores;
- `tensor.empty`, constants and scalar extraction;
- `linalg.fill` and `linalg.generic` with projected-permutation/broadcast indexing maps, parallel iterators and bounded sum reductions;
- scalar add/subtract/multiply/divide/negation, sqrt/sin/cos/tanh, and the integer casts/comparisons needed by sparse output gathers.

There is no need to implement general matrix multiplication, convolution, dynamic shapes, custom calls, or arbitrary affine maps to obtain the measured small-program benefit. Named contractions and long reductions keep IREE or Accelerate. The extra tensor-pad/slice/reshape code in the exploratory translator was for the rejected whole-program probe and need not enter the first implementation.

| Function | All tensors ≤256: helpers/sites replaced | Output ≤256: sites | Output ≤64: sites | Large outer loops: sites |
|---|---:|---:|---:|---:|
| chain_jac M=3 | 6/12 | 30 | 2 | 83 |
| chain M=3 | 16/40 | 82 | 6 | 180 |
| race_cars N=5 | 123/229 | 229 | 216 | 229 |
| race_cars N=500 | 1/1 | 5 | 5 | 229 |
| npmpc N=12 | 13/13 | 44 | 25 | 61 |


Production risks are exact float and integer semantics, reductions and uninitialized `tensor.empty` values, source/destination overlap across multiple results, signed zero, NaN/infinity, integer overflow/casts, generated-code growth and LLVM dialect-version coupling. The prototype checks finite workload inputs; it is not a production compiler. It must not introduce unsafe `0*x -> 0` folding for floating inputs. The final prototype removes that simplification; earlier preliminary timings were superseded. Safe tensor-view composition and common subexpression naming are enough for the reported final measurements.

A strict eligibility budget is preferable to a new general compiler. Estimate 300-500 focused production lines plus meaningful semantic tests for the initial subset, with explicit fallback on everything else. The current prototype is about that size including experiment plumbing and the whole-program probe. The host change itself is small and can follow the existing BLAS replacement route in `_Host.dispatch`.

## Implementation plan for one agent

1. Add a small-dimension lowering-configuration pass. Use `tune_configs.py`'s `power2small` predicate, not a benchmark table. First implement it as a controlled split compilation or an equivalent Transform-dialect matching specification, preserving existing host, HAL layout, BLAS selection and fallback. Estimate 80-150 focused lines plus checks. Gate on static extents and supported CPU tiling levels; leave non-power-of-two extents and oversized virtual vectors alone. Run ordinary pytest/lint/type checks, finite and nonfinite numerical tests, and these five interleaved workloads. This phase has no dependency on item J or on C codegen.
2. Add an optional bounded float64 C emitter through the existing `_Host.dispatch` replacement route. Start with the subset above and a strict all-tensor limit, scalar-operation/code-size budgets, static offsets and small reductions. Reuse HAL's allocation, aliasing and input-staging analysis. Keep IREE's original dispatch for anything unsupported, including large products and unknown indexing. Estimate 300-500 production lines plus tests. Test zero/NaN/infinity semantics, reductions, offsets, aliasing and multiple outputs before considering it a default. Expected gain is roughly 26% on the N=5 race kernel, approximately zero on N=500 or npmpc, and no demonstrated chain gain. This phase is independent of phase 1; measure their combination before claiming combined gains.
3. Keep `fuse-multi-use=false` as a measured compiler-policy candidate, rather than flipping it globally. It reduces chain_jac latency by 12% here, but has an approximately 18% N=5 penalty in the broad run. A future producer-cost rule should account for duplicated operations and broadcasts. Do not select it by workload name. This is a separate optimization from scalarization and can be deferred.
4. After item J supplies local stage derivatives, repeat this experiment on those local functions. That is the correct point to ask whether a whole stage fits the tiny-C budget. Preserving or explicitly exposing pure local tensor functions before IREE flattens them has more potential than translating a large asdex graph after dispatch formation. A generic whole-program scalarizer that also tracks pad/slice/select dependencies and proves safe indexing is a larger compiler project, roughly 1,000 or more lines plus sustained semantic maintenance. The present measurements do not justify starting it.

Per-workload expectation from the measured compiler mechanisms: chain_jac gets a possible approximately 12% improvement from fusion policy, chain Hessian only a few percent, tiny race gets approximately 26% from bounded C, large race gets approximately 16% from tiles, and npmpc has no established gain beyond measurement noise. These are individual mechanisms, not an unmeasured composite speedup. Even the best candidate remains several times slower than Scaly on the dynamics workloads.

Stop after these bounded phases unless fresh evidence changes that decision. Do not implement the large-loop C route, unit vector tiles, unconditional short-reduction expansion or global single-dispatch grouping. Keep the rejected-mechanism measurements so the next agent need not rediscover them.

## Reproduction and artifacts

The workspace remains unchanged according to `git status --short`. Run from the supplied aotax worktree. Every generated library and source goes under `/tmp/aotax-k`.

```sh
cd /Users/tudoroancea/.t3/worktrees/aotax/t3-7c33bedc
# All builds, checks, five-process comparisons and profiles:
/tmp/aotax-k/reproduce.sh

# Fast inspection of already-built binaries, with original-input checks:
uv run --group bench python /tmp/aotax-k/time_all.py --output repeat

# Independent perturbed-input checks:
uv run --group bench python /tmp/aotax-k/verify_native.py \
  chain_jac-3 chain-3 race_cars-5 race_cars-500 npmpc-12
```

Primary artifacts:

- `final-times.json`, `final-times-best.json`: the broad five-process interleaved sweep.
- `confirmation-times.json`, if present: the final focused confirmation including power-of-two tiles. Its matching baseline must be used for its ratios.
- `measurements.md`, `measurements.json`: machine-readable and Markdown summaries.
- `<case>/<variant>/bench`: native binary; every invocation checks all outputs.
- `<case>/base/iree/{stablehlo.mlir,hal.mlir,host.c,sources/}`: unchanged aotax compilation.
- `<case>/configs/`, `<case>/fullsmall/`, `<case>/power2small/`: concrete lowering configs and compiled variants.
- `<case>/llvm/`: code-generation LLVM, optimized LLVM, assembly and object files.
- `<case>/profile/{profile.txt,profile.json}`, `dispatch-summary.json`, `llvm-loops.json`: counts, dimensions and attribution.
- `<case>/scaly/generated.c`, `scaly-summary.json`: Scaly code and lowered-program statistics.
- `<case>/verification.json`: perturbed-input checks and worst absolute errors.
- `study.py`, `prepass.py`, `tiny_codegen.py`, `tune_configs.py`, `one_dispatch.py`: mechanisms.
- `iree-help.txt`: the pinned compiler's actual available options.

The report does not propose repeating rejected vector-peeling, tanh approximation, broad transcendental barriers, native 32/64-byte vector widths, SME, aggressive fusion, or inline-static experiments. It does not change the derivative algorithm or claim to close its arithmetic gap.
