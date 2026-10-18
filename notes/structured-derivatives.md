# Structured sparse derivatives for aotax

Design experiment, 18 October 2026, on main at ce1fc4d. Its code, generated libraries, logs and measurements were written to `/tmp/aotax-j`, outside the repository, and are not kept; the paths below refer to that scratch directory.

## Recommendation

Build a JAX-to-JAX derivative transform around **local dependency graphs and shared computations**. Recover small computations and their static input maps from the program, differentiate them with JAX, and assemble only the requested sparse entries. Use partial separability as a particularly cheap case of this more general graph transformation. Keep matrix products as matrix products. Within small non-matrix computations, allow scalar leaves and dead-code elimination (DCE) before differentiation, so JAX can remove absent tangents. Keep asdex as the whole-function fallback until the extraction proof covers a program.

The representation should support a graph of local maps, not only a list of independent stage costs. For unbumpercars, a car's neural step feeds several pair factors. Compute that step and its derivatives once per car, then apply the chain rule to the pair and wall factors. Flattening each pair into a separate function of its controls would repeat the expensive network. For chain, the shooting stage is too coarse: retain the smaller mass neighborhoods inside it.

This takes inspiration from Scaly without adopting its representation or differentiation algorithm. In particular, the proposed transform can find repeated dependency neighborhoods after `vmap` has disappeared, can compose several levels of local maps, and can choose dense, colored or scalar automatic differentiation (AD) independently for different pieces. There should be no model predictive control (MPC), car, horizon or benchmark names in the transform.

The experiments support this architecture and show substantial native speedups. They do **not** establish a finished automatic optimizer that attains Scaly's operation count on every workload. The distinction between automatic extraction, explicit mathematical factoring, and derivative execution is recorded below. The large-chain Hessian remains the most consequential unresolved case.

## What was read

The supplied standing instructions; the checkout's `AGENTS.md`, `PLAN.md` F2 and Phase 3 J, `README.md` benchmark section, `benchmarks/README.md`, `benchmarks/workload.py`, `benchmarks/harness.py`, the public and private formulations for chain, race_cars, npmpc, npmpc_decoder and unbumpercars, and the compilation, host, math, Basic Linear Algebra Subprograms (BLAS), custom-call and Metal paths in `src/aotax/` informed this note.

Scaly sources inspected were `/Users/tudoroancea/dev/scaly/src/scaly/ad/{sparse,derivatives,forward,reverse,sparsity}.py`, its lowering and program-pass pipeline, and `codegen/{aot,c}.py`. Measurements use the Scaly version pinned in this checkout's `uv.lock`, through the existing benchmark helpers. `environment.json` records versions and revisions.

## Evidence and boundaries

There are three separate prototypes.

* `analyze.py` propagates structural input dependencies through the unchanged whole-program Jaxprs, using asdex's primitive propagation. It checks candidate array axes and reports bounded local supports, affine strides and overlaps. `provenance.json` contains the results. It finds shooting axes for chain and race cars, mixed state/control strides for npmpc, and both car and pair dependencies for unbumpercars. It does not yet extract executable local array programs. A bounded dependency set is a candidate for extraction, not by itself a proof that an arbitrary operation can be unbatched.
* `automatic.py`, `auto_blocks.py` and `scalarize.py` implement source-independent scalar subgraph extraction. They replay a supported small Jaxpr with scalar input leaves, use JAX's automatic differentiation, remove dead code for each output or gradient component, canonicalize the surviving Jaxprs, group repeated computations, and evaluate each group with `vmap`. No stage dimensions or input neighborhoods are supplied. The unchanged race-car Jacobian and Hessian at N=5 and chain Jacobian at M=3 pass the independent dense references. `automatic.json` records the groups, costs and construction time. This is a working generality experiment, not the scalable extractor to ship.
* `structured.py`, `factors.py` and `neighborhoods.py` implement and measure the intended execution layer at every default size. They use unchanged benchmark stage functions but explicitly factor the outer Lagrangian/constraint assembly. `neighborhoods.py` discovers scalar output supports and repeated programs inside a chain step. `factors.py` supplies the shooting layout and the shared-map chain rule for unbumpercars. These are mathematical rewrites under `/tmp`, not benchmark-specific cases in an automatic production utility. The main performance tables measure this execution layer, not an already completed zero-rewrite front end.

`generic_checks.py` checks a windowed sine energy, repeated gather indices, and a squared global sum against dense Hessians. The last case is deliberately dense: summing independent terms and then applying a nonlinear operation destroys sparse partial separability. The prototype returns the dense lower pattern rather than pretending that the terms are still independent.

All main benchmark binaries use `benchmarks/harness.py` and its C++ correctness gate. Before timing, the gate checks the stored entries and every omitted matrix entry against the independent dense reference, at `1e-9 + 1e-9 * abs(reference)`. Hessians check the lower triangle. The chain reference uses plain dense stage `jax.hessian`/`jax.jacfwd` and an independent assembly, as established in the repository. The other benchmark reference functions use whole-program dense JAX differentiation. Build-time checks also compare the JAX prototype before aotax compilation. The selected method for every default size also passed two additional perturbed cases in both JAX and the native library, including zero objective/constraint multipliers where those are inputs. All 96 comparisons passed; the largest absolute error was `9.1e-13`. `verification/*.json` records each comparison.

The npmpc_decoder width sweep was read for context but was not part of the requested four-family timing experiment. It should be added as a contraction-preservation regression before implementation ships.

The all-size performance prototype includes no source rewrites in the checkout. The natural rewrites in `factors.py` are:

* Chain: move the dynamics multiplier dot product inside the stage map, add the existing stage cost there, and keep the terminal cost separate. Jacobian assembly adds the initial identity and following-state negative identities.
* Race cars: separate weighted dynamics, node costs/corridor terms, and the affine initial/following-state terms. All nonlinear functions come from `_race_cars.py`.
* Npmpc: separate the existing decoder dynamics, quadratic interstage cost, terminal cost and slack penalty. The local decoder input has five entries, not the whole decision vector. Following-state and bound Jacobians are affine maps. Runtime cost weights remain runtime inputs, including the entries whose sampled values happen to be zero.
* Unbumpercars: explicitly compose the existing per-car step with pair and wall barriers. Only step outputs 3:6 affect the downstream barrier derivatives with respect to controls. The pose is independent of controls and the downstream barriers ignore the last steering state. The local Jacobian/Hessian chain rule therefore uses three active step outputs and two controls per car. These dependencies are properties of this formulation, not assumptions a general utility may make without proving them.

The generic assembler sorts and coalesces coordinate pairs. It lowers repeated contributions to a static padded gather followed by addition, rather than runtime sparse matrices or atomics. It creates only a values vector at runtime. The tested local maps use unique input coordinates. A production extractor must deduplicate a repeated input gather before differentiating; otherwise a local off-diagonal that maps to a global diagonal needs both symmetric contributions. The source-independent repeated-gather check exercises deduplication through scalar input leaves.

## Measurements

<!-- MEASUREMENTS START -->
| Workload | Size | Structured method | Operations, millions: asdex / structured / Scaly | CPU microseconds: asdex / structured / Scaly | asdex / structured speedup |
|---|---:|---|---|---|---:|
| chain | 3 | scalar | 0.771 / 1.413 / 0.218 | 86.22 / 12.71 / 22.76 | 6.79x |
| chain | 9 | colored | 8.324 / 8.072 / 1.928 | 1051.49 / 1030.16 / 336.54 | 1.02x |
| chain | 17 | colored | 19.585 / 19.001 / 4.351 | 2421.84 / 2366.62 / 1147.54 | 1.02x |
| chain_jac | 3 | neighborhood | 0.493 / 0.280 / 0.068 | 63.56 / 4.76 / 3.91 | 13.34x |
| chain_jac | 9 | neighborhood | 3.140 / 4.247 / 0.516 | 354.41 / 77.31 / 61.31 | 4.58x |
| chain_jac | 17 | neighborhood | 10.925 / 9.617 / 6.172 | 1664.91 / 182.61 / 400.53 | 9.12x |
| npmpc | 12 | dense | 0.600 / 0.444 / 0.350 | 16.23 / 14.07 / 15.14 | 1.15x |
| npmpc | 50 | dense | 2.496 / 1.846 / 1.450 | 57.96 / 46.66 / 61.96 | 1.24x |
| npmpc | 200 | dense | 9.982 / 7.382 / 5.793 | 199.58 / 173.93 / 249.19 | 1.15x |
| npmpc_jac | 12 | dense | 0.406 / 0.217 / 0.191 | 8.99 / 6.99 / 9.74 | 1.29x |
| npmpc_jac | 50 | dense | 1.692 / 0.905 / 0.789 | 32.52 / 28.26 / 40.60 | 1.15x |
| npmpc_jac | 200 | dense | 6.769 / 3.619 / 3.151 | 124.70 / 99.05 / 163.03 | 1.26x |
| race_cars | 5 | scalar | 0.043 / 0.013 / 0.006 | 3.51 / 1.22 / 0.67 | 2.87x |
| race_cars | 40 | scalar | 0.342 / 0.102 / 0.045 | 17.63 / 4.28 / 3.89 | 4.11x |
| race_cars | 500 | scalar | 4.266 / 1.271 / 0.558 | 252.94 / 55.30 / 48.52 | 4.57x |
| race_cars_jac | 5 | scalar | 0.026 / 0.006 / 0.002 | 2.30 / 0.77 / 0.45 | 2.98x |
| race_cars_jac | 40 | scalar | 0.210 / 0.044 / 0.018 | 14.63 / 3.05 / 3.04 | 4.80x |
| race_cars_jac | 500 | scalar | 2.623 / 0.550 / 0.228 | 204.88 / 40.25 / 41.01 | 5.09x |
| unbumpercars | 2 | dense | 1.756 / 1.138 / 1.978 | 45.21 / 26.59 / 147.63 | 1.70x |
| unbumpercars | 4 | dense | 5.913 / 2.299 / 3.678 | 86.04 / 32.93 / 276.83 | 2.61x |
| unbumpercars | 8 | dense | 21.643 / 4.717 / 7.506 | 244.70 / 60.79 / 574.71 | 4.03x |
| unbumpercars_jac | 2 | dense | 1.006 / 0.566 / 0.564 | 26.28 / 16.59 / 41.33 | 1.58x |
| unbumpercars_jac | 4 | dense | 2.308 / 1.128 / 1.125 | 30.13 / 20.23 / 83.24 | 1.49x |
| unbumpercars_jac | 8 | dense | 6.971 / 2.256 / 2.261 | 69.37 / 31.64 / 174.10 | 2.19x |
<!-- MEASUREMENTS END -->

A table entry's arithmetic is the optimized XLA `cost_analysis()['flops']` estimate for the JAX derivative, including assembly. Scaly's number counts floating arithmetic nodes in its lowered scalar program, expands constant loop trip counts and follows calls, using the same approach as F2. Transcendentals are recorded separately in JSON. These are different compiler-level models, not retired machine instructions. They are particularly poor predictors for the scalar-leaf candidates: IREE optimizes and vectorizes those programs differently from XLA. The measurements therefore answer both the requested arithmetic question and the independent native runtime question; a faster native program does not establish a lower arithmetic count.

Timing uses native Google Benchmark CPU time in fresh processes, excludes construction/compilation/context creation, interleaves the available variants in alternating order, and takes the best of five. Raw repetitions are in `results.json`. The machine is shared with other experiments. Do not compare initial build-time smoke timings with the final interleaved values.

In the method labels, `dense` means small local dense derivatives, with shared-map composition for unbumpercars. It does not mean a whole-problem dense derivative. `scalar` uses scalar local leaves, `colored` uses local asdex coloring, and `neighborhood` discovers repeated scalar-output neighborhoods inside a chain step.

The selected structured column is an experimental choice among the measured methods, documented per row in the generated tables. No automatic method selector or autotuner has been implemented. The selected family policy was fixed before the final interleaved run; it is not a per-repetition choice. The artifact retains unsuccessful candidates as evidence, including full-stage dense blocks and local coloring. Use a performance acceptance test before promoting any selection rule.

## Why these approaches differ

| Approach | Generality and exactness | Implementation cost | What the evidence says |
|---|---|---|---|
| Array-axis analysis and local extraction | Covers elementwise maps, static windows/gathers, reductions, batched contractions and nested local maps when primitive rules prove the input/output relation. Overlapping input windows are allowed. Exact up to ordinary floating-point evaluation differences. | An abstract interpreter plus backward demand/extraction. Must handle axes independently of their numeric length and preserve shared subgraphs. | The provenance probe finds the needed axes in all four unchanged programs. Whole-stage dense differentiation alone leaves substantial excess work, particularly for chain. |
| Additive partial separability | For `f = sum_i phi_i(x[S_i], p)`, `H = sum_i P_i.T @ H_i @ P_i`. Static gather maps and overlaps work. Parameters and multipliers may vary per factor. | The smallest useful Hessian implementation once extraction exists. Do not assume a nonlinear function of a sum is separable. | Good for MPC Lagrangians. Neural dynamics, costs, affine constraints and terminal terms need distinct local units to avoid dense zero tangents. |
| Shared-map composition | Generalizes independent factors to a graph. For `f = phi(T(x))`, retain `J_T.T H_phi J_T + sum_k grad_phi[k] H_T[k]`. | More work than a flat factor list, but only local dense products and static assembly are needed initially. | The unbumpercars prototype avoids repeating networks for pairs. This is essential to the recommendation, even if it follows the first usable milestone. |
| Improved global coloring | Keeps asdex's broad primitive/control-flow coverage. Direct recovery into the lower triangle changes output work, not the color products. Structurally sparse tangents can remove arithmetic but need their own storage/routing rules. | Direct triangle recovery is small. General packed tangent propagation is another substantial interpreter. Local coloring can reuse asdex. | Local coloring gives only modest Hessian arithmetic savings for chain. Direct lower recovery leaves the F2 arithmetic count unchanged. Global color count alone is not the source of the whole gap. |
| Scalar subgraph AD using JAX rules | Does not require explicit source maps. DCE finds each scalar output's true input subgraph and can group repeated neighborhoods. Exact on the supported primitive subset. | Small prototype, but whole-program expansion and repeated row traces can explode. Must retain matrix operations and apply it only to bounded local programs. | Correct automatic small cases and strong native scalar-local results. Whole-program chain Hessian construction exceeded 13 GB of resident memory and was stopped. The naive M=9 scalar Hessian also became too expensive to construct, so it is not a scalable front end. |
| General graph elimination/edge pushing | Can exploit sparsity with no stage abstraction and choose forward/reverse elimination. Symmetry can reduce Hessian work. | A larger AD/graph-optimization project with fill-in, ordering, storage and primitive-coverage questions. | Worth a later experiment, especially within difficult local blocks. There is no measured evidence here that a new full elimination engine would beat the simpler hybrid. |

The core correctness identity for a local map is broader than independent batch axes. A local factor may read several windows, share a global scalar, or read arbitrary compile-time gather indices. Independence means that the extracted computation has no unaccounted dependencies, not that two factors must read disjoint inputs.

For a Jacobian, each output block has its own input gather and output coordinate map. Differentiate the local function and route those entries directly to the global matrix. For a Hessian of a sum, assemble local lower triangles and add overlaps. For composition, propagate local adjoints as well as local Jacobians and weighted Hessians. This allows several pair factors to reuse one expensive map. Affine maps need only index transformations and constants, and contribute no second derivatives.

There is also a more general Hessian entry point: analyze the Jaxpr of `jax.grad(f)` and extract the small input neighborhoods of its output blocks, then compute their Jacobians. This does not require recognizing an explicit sum in the primal function. A nonlinear global reduction naturally produces global gradient dependencies instead of a false block decomposition. The automatic scalar prototype exercises this idea. For production, do the extraction on compact arrays before scalar expansion, preserve shared work across gradient blocks, and keep the primal-map route where it gives cheaper weighted-Hessian composition. This is a candidate implementation simplification, not a measured all-size result.

The derivative method should be a decision within each local program. Use dense JAX AD for genuinely small dense blocks and neural contractions. Use asdex coloring when the local block is sparse but too large to scalarize. Use scalar input leaves and subgraph DCE when this removes substantial inactive arithmetic in a small non-matrix program. Start with a deterministic, documented policy based on graph properties; native measurements must validate it. XLA's cost analysis cannot be the only policy input.

## Existing work and what carries over

[sparsejac](https://github.com/mfschubert/sparsejac) takes a supplied sparsity pattern and uses compressed forward/reverse differentiation. It does not recover repeated local computations. Improving global coloring alone leaves the dense intermediate color batches seen in F2.

[JAX's experimental sparse module](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) represents and transforms sparse array computations. Sparse-aware differentiation differentiates stored sparse values in a supplied sparse subspace. It is not an extractor of sparse derivatives from arbitrary dense-array programs. Its documentation describes the implementations as reference code and discourages performance-critical use. No runtime sparse-array representation is needed for a static values/coordinates output.

[Graphax](https://github.com/jamielohoff/graphax) is closer to the more general alternative. Its `jacve` performs sparse cross-country elimination, accepts an elimination order and composes with JAX transformations. That is useful prior work for a future local elimination method, and a reason to avoid immediately writing another complete AD engine. Compatibility, primitive coverage and aotax/IREE performance on these exact workloads were not tested.

[Bell and Kristensen's subgraph methods](https://arxiv.org/abs/2111.05207) compute sparse Jacobians/Hessians without coloring or compression. This motivates the output-demand/DCE experiment here. Their results do not establish performance for JAX or IREE. [Gower and Mello's framework](https://arxiv.org/abs/2007.15040) develops reverse Hessian computation and edge pushing with symmetry. It is a broader algorithmic option, with fill-in and ordering costs that this spike has not measured.

[Hill and Dalle's sparsity-detection work](https://arxiv.org/abs/2501.17737) separates dependency detection, coloring and differentiation. Reuse that separation: proven structure, a derivative execution choice, and sparse recovery should be distinct. The pinned asdex detector is useful immediately, while compact affine/local maps should replace large per-element sets where needed for construction scale.

[Pallas](https://docs.jax.dev/en/latest/pallas/) is a GPU/TPU kernel language, not a solution to finding derivative structure in the CPU JAX functions in this task. It may help a later GPU execution layer after local structure is known. It does not solve the present AD arithmetic issue. [JAX's interpreter guide](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html) provides the general interpreter pattern, but internal Jaxpr and DCE APIs still require version-specific tests.

Scaly's inspected `sparse_jacobian` splits concatenations of VMAP expressions, colors small per-formal patterns, specializes constant seeds, and maps the resulting derivatives. Other dependent pieces can fall back to global coloring. Its `sparse_hessian` first builds/simplifies the scalar gradient and then uses a global star coloring, including recovery from either symmetric orientation. Its compiler then simplifies/scalarizes small computations, fuses loops and emits scalar C. Therefore the F2 result is not evidence that Scaly computes every Hessian from explicitly assembled local Hessians. What carries over is constant specialization, skipping absent tangents, preserving repetition, direct triangle recovery and inspecting the generated work. The proposed hierarchical local graph is a separate design.

## Proposed interface

Keep the experiment in a small `aotax.sparse` module initially. Do not create a separately versioned package until an independent JAX consumer needs it. Expose an optional sparse dependency extra for asdex; importing ordinary aotax should not require the extra.

```python
from aotax.sparse import sparse_hessian, sparse_jacobian

H = sparse_hessian(lagrangian, *specs, argnums=0, triangle="lower")
J = sparse_jacobian(constraints, *jac_specs, argnums=0)

lib = aotax.compile(H.values, *specs, out="build/hessian")
# H.rows, H.cols, H.shape are immutable NumPy metadata.
# H.report explains the extracted maps and any fallback.
```

Return one small immutable result with `values`, `rows`, `cols`, `shape`, and `report`. Runtime returns only the one-dimensional values array. Coordinates are sorted, unique and stable for the supplied input shapes/dtypes and static configuration. Retain structural entries even when the sampled value is zero. A structural pattern may conservatively include zero entries; it must never omit a potentially nonzero derivative.

For the first version, support one real floating array selected by integer `argnums`, scalar Hessian outputs and one array Jacobian output. Flatten selected input/output arrays in documented row-major order. Multiple selected arguments, pytrees and complex derivatives can follow after the core path is reliable. The specifications provide shapes/dtypes, not numerical evidence of zero structure. Runtime parameters, objective multipliers and constraint multipliers remain inputs.

Provide `method="auto" | "structured" | "asdex" | "dense"`. `auto` uses only proven extraction rules and otherwise falls back. `structured` raises an actionable unsupported-structure error instead of silently falling back. `asdex` reproduces the baseline. `dense` is a small-problem reference/last fallback. The report should include recognized repetitions and local widths, affine maps, shared subgraphs retained, derivative methods, estimated temporary sizes, coordinates before/after coalescing, and the precise operation/reason for fallback. Do not catch arbitrary compiler or correctness errors and call them unsupported structure.

The mathematical guarantee is the derivative defined by JAX's rules on the supported domain, including custom derivative rules when supported. It is not bitwise equality between different evaluation orders. Existing aotax compilation approximations, flush-to-zero behavior and floating-point limits still apply. Nonsmooth operations follow JAX's chosen derivative. Skipping structural zero tangents does not grant permission to change NaN, infinity or signed-zero behavior without a stated domain contract.

A Jacobian-vector product (JVP) and a vector-Jacobian product (VJP) may have user-defined rules in JAX. The current scalar replay inlines the benchmark's custom tanh primal. Its supplied derivative rule is the same `1 - tanh(x)**2` rule, and the outputs are verified numerically. This does **not** justify inlining arbitrary `custom_jvp` or `custom_vjp` calls. Production must preserve their rules or fall back. The prototype's primitive subset and whole-program memory growth are not appropriate as a general public guarantee.

The benchmark change should add an optional structured builder to `Workload` and a backend named `aotax_structured`. Leave `aotax` and `scaly` builders, CSV names and correctness checks unchanged. An initial implementation can return `H.values, H.rows, H.cols` from that builder. Add the extra column only when requested or present. Write the report and selected method beside the library so a fast number is traceable to the actual path.

## Implementation plan

1. **Usable common output and baseline.** One agent can add the result object, lazy asdex dependency, reference dense route, explicit method selection/report, and direct lower-triangle recovery. Add meaningful exactness cases for windows, overlaps, duplicate gathers, varying runtime parameters, nonlinear outer reductions and custom-derivative fallback. Add the second benchmark column. This milestone is usable for all current functions through fallback, without claiming structured speedups. Keep all current baseline columns intact.
2. **Automatic independent maps and sums.** Implement compact input-index provenance and dependency-safe axis rules for elementwise operations, reshape/transpose, static slice/gather/concatenate, reductions and batched dot. Separate affine terms and scalar sums/weighted reductions. Perform backward demand extraction to make one small local Jaxpr, then use JAX AD and static sparse assembly. Verify primitive rules on generated small programs against whole dense AD. Ship automatic chain/race/npmpc map/sum paths, with unrecognized portions using the whole-function fallback. Do not expose workload-specific stage callbacks as the public solution. This depends on phase 1.
3. **Bounded local AD and nested neighborhoods.** Move scalar-leaf/DCE specialization inside extracted local non-matrix computations, with limits on graph size and temporary memory. Retain matrix contractions. Recover nested mass neighborhoods and group local Jaxprs by normalized input maps, including parameter windows. Compare dense, colored and specialized local paths on the full default sweep. The large-chain Hessian needs an explicit acceptance result before declaring Phase 3 J finished. This depends on phase 2, although the local-specialization experiment can be developed independently against a small internal local-program interface.
4. **Shared local maps.** Retain map nodes that feed several factors, and implement local Jacobian/weighted-Hessian composition. Use the unbumpercars prototype as a reference for chain-rule values, not as hard-coded implementation logic. Require that the network executes once per car, and test arbitrary shared-map examples. This depends on the extraction/assembly interface from phase 2; it can be implemented alongside phase 3 with separate file ownership if agents are explicitly assigned.
5. **Control flow and broader coverage.** Support `scan` only when carry analysis proves a bounded local relation or an affine propagation that the representation handles. An actual nonlinear rollout carry couples stages and may produce dense derivatives. Keep general scans, dynamic gathers and unknown custom derivatives on fallback. Add them only with sound rules and independent tests. Consider local graph elimination/Graphax only after profiling a remaining local AD bottleneck.

Each phase should leave a functional end-to-end path. Avoid a large general AD framework before the map/sum path ships. No separate custom C compiler, solver API, new sparse runtime or global autotuning service is needed for the first version.

## Risks and remaining questions

The main unproven part is scalable automatic extraction of the execution patterns measured here. The provenance probe and small automatic subgraph examples demonstrate that the information is present; the all-size measurements demonstrate the execution benefit. The bridge between them still needs implementation and soundness tests.

Recovering maps by shape alone is unsafe. A reduction or contraction may mix an apparent batch axis. Static input windows may overlap. A broadcast global variable selected for differentiation couples every factor even though its parameter value is shared. Multiple appearances of one decision coordinate must be deduplicated. These are correctness rules, not heuristic pattern matches.

Constructor time is part of usability. The stopped scalar-chain attempts show why whole-function expansion, repeated complete gradient traces, per-output scans over a huge Jaxpr and dense per-element provenance must be bounded. Extract array-local programs before scalar specialization, and cache normalized local programs. Parameter-window lifting is necessary to merge repeated factors whose parameter slices have different global indices.

A padded gather reducer is simple but can waste space when contribution counts vary widely, and `Assembly.finish` in the spike scans coordinate groups repeatedly. Production should build contribution lists in one pass and bucket by count or use a fixed segmented reduction. Non-overlapping blocks can write directly. Correctness should not depend on floating-point reassociation hiding duplicate coordinates.

Small contraction shapes in the first unbumpercars chain-rule formulation triggered IREE initialized integer globals, which the current aotax host rejects. Rewriting those tiny products as explicit fixed-size sums allowed compilation without repository changes. Record this as a lowering limitation to test. Large neural matrix products remain intact.

XLA cost estimates and IREE native behavior can disagree strongly. Measure native time, allocations, dispatches and construction memory in addition to arithmetic estimates. A default method policy must explain its choice and avoid promising that every recognized structure is faster. Keep baseline and forced-method options available for diagnosis.

Numerical equivalence has been checked at the benchmark inputs, two perturbed cases per default size, and generic tests, not proven for every possible input. Production needs broader randomized weights and scales, supported nonsmooth choices, custom rules and nonfinite behavior. Changes in summation order and existing aotax math approximations can affect tolerances.

## Reproduction and artifact map

The complete sequence is also available as `zsh /tmp/aotax-j/reproduce.sh`. It excludes the deliberately stopped large scalar-Hessian experiments.

Run from the aotax checkout used for this experiment, or set `AOTAX_ROOT` to it. All generated outputs stay under `/tmp/aotax-j`.

```sh
export AOTAX_ROOT="/Users/tudoroancea/.t3/worktrees/aotax/t3-55c62f49"
export UV_PROJECT_ENVIRONMENT=/tmp/aotax-j/.venv
export SCALY_CACHE_DIR=/tmp/aotax-j/scaly-cache
export PYTHONPYCACHEPREFIX=/tmp/aotax-j/pycache
cd "$AOTAX_ROOT"
uv sync --locked --group bench
uv run --no-sync python /tmp/aotax-j/analyze.py
uv run --no-sync python /tmp/aotax-j/check.py
uv run --no-sync python /tmp/aotax-j/check_automatic.py
uv run --no-sync python /tmp/aotax-j/generic_checks.py
uv run --no-sync python /tmp/aotax-j/bench.py
zsh /tmp/aotax-j/variants.sh
uv run --no-sync python /tmp/aotax-j/bench.py --name chain --size 3 --modes scalar
for size in 3 9 17; do
  uv run --no-sync python /tmp/aotax-j/bench.py --name chain_jac --size "$size" --modes neighborhood
done
uv run --no-sync python /tmp/aotax-j/bench.py --name chain --size 17 --modes direct_lower
uv run --no-sync python /tmp/aotax-j/bench.py --timing --modes asdex,dense,scaly,scalar,colored,neighborhood,direct_lower --reps 5
uv run --no-sync python /tmp/aotax-j/verify.py
uv run --no-sync python /tmp/aotax-j/make_report.py
```

`bench.py` reads the already installed Google Benchmark prefix at `/Users/tudoroancea/dev/aotax/benchmarks/build/third_party/gbench`. Override the `harness.gbench` binding for another installation. This was the only reused build dependency. It does not write to that prefix. The first run builds binaries; repeated runs reuse completed `meta.json` records. Incomplete stopped-experiment directories have no successful metadata and are excluded. To rebuild a completed variant, remove only its directory under `/tmp/aotax-j/build`.

Do not run `scalar-chain.sh` or `check_automatic.py --large-chain` as part of routine reproduction. These preserve the expensive failed experiments for deliberate investigation. `stopped-experiments.json` records why they were stopped and explicitly has no invented runtime result.

The main artifacts are `results.json` with all final timing repetitions and per-variant costs, `summary.json`/`table.md` with the selected comparison, `automatic.json` with source-independent extraction checks, `provenance.json` with unchanged-program analysis, `verification/*.json` with perturbed JAX/native comparisons, and `build/<name>/N<size>/<method>/` with source/intermediate representations, metadata and correctness-gated binaries. `build.log`, `build-resume.log`, `variants.log` and the individual candidate logs retain construction results. `environment.json` records the machine and exact versions.


## Lessons that also apply to Scaly

This section records transferable observations, including negative results. It proposes no changes to the Scaly checkout. Numerical claims refer to this experiment's artifacts, and proposed changes still need a Scaly-specific test.

1. A low color count does not guarantee low work. Dense tangent batches can carry zeros through most intermediate operations. Scaly already specializes constant seeds; inspect whether each local intermediate still evaluates inactive directions. Measure the generated work, not only the number of colors.
2. Choose the differentiation method locally. A scalar output with a small dependency neighborhood can favor reverse AD; a neural map with a few active controls can favor forward AD; a large sparse local Hessian may still favor star coloring. One global differentiation choice need not be optimal for a heterogeneous expression graph.
3. A shooting interval can be too coarse a local unit. Chain has locality both across time and across masses. Recover and retain the inner neighborhoods instead of treating a whole large chain state as dense. This observation applies to spatial discretizations and graph models as well as MPC.
4. Preserve a shared expensive map across downstream factors. Unbumpercars computes one network per car, then uses it in several pair/wall factors. Rebuilding that network for each pair loses the benefit of local differentiation. A graph of reusable maps is more general than a flat list of separable factors.
5. Compute only derivative outputs that downstream consumers use. The unbumpercars barriers need three control-dependent step outputs, even though the step returns seven states. Output liveness can reduce Jacobians and second derivatives before any coloring. The reduction must follow proved dependencies, not the meaning of the state names.
6. Weighted local Hessians may be enough. In second-order composition, a downstream adjoint contracts the map's output Hessians. Once that adjoint is available, investigate computing a weighted Hessian directly instead of every output Hessian. The current prototype computes all three active output Hessians; that leaves a possible further improvement. Reuse the map's primal residuals where practical.
7. Separate affine routing from nonlinear differentiation. Initial-state identities, following-state negative identities, slack appearances and many constraint terms need index maps and constants. They do not need tangent products, and affine terms contribute no Hessian. Keep these pieces visible through lowering.
8. Runtime coefficients are not structural constants. Several sampled npmpc cost weights are zero, but they remain runtime inputs. Sparsity and optimizations must remain valid when they become nonzero. Test changed weights and multipliers, not only the standard sample.
9. Build lower-triangle recovery directly. A full sparse decompression followed by a lower selection needlessly creates upper entries. Scaly already retains recovery metadata for triangle selection; preserve that direct recovery through simplification and lowering. It changes output traffic, not the cost of the color products.
10. Overlapping contributions should have a compile-time assembly schedule. Constant gathers and fixed small sums can replace runtime scatter-add and atomics. Separate non-overlapping output blocks so they write directly. Bucket differing contribution counts rather than padding every entry to the largest count.
11. Repeated input indices require special care in Hessian assembly. Deduplicate local decision coordinates before differentiation. If different local slots map to the same global coordinate, the two orientations of an off-diagonal local Hessian contribute to a global diagonal. Merely sorting `(row, col)` and summing one local triangle can miss a factor of two.
12. Repeated derivative programs can be discovered from dependency subgraphs, even when no explicit VMAP survives. DCE extracts the inputs needed for an output or gradient component. Canonical local programs can then be batched. For Scaly, this suggests an optimization beyond preserving VMAP nodes supplied by the user.
13. Normalize parameter windows as well as decision windows when grouping repetitions. Two otherwise identical local programs can appear different solely because their parameter slices have different global offsets. Lift those slices into explicit per-instance arguments. The scalar automatic experiment exposes this missed grouping for race-car corridor terms and weighted Hessian rows.
14. Keep contractions intact when scalarizing. The network workloads already benefit from dense matrix libraries. Scalarizing a whole model to expose tiny nonlinear derivatives can destroy useful matrix structure or create huge source graphs. Apply scalar specialization only inside bounded non-matrix local computations.
15. Scalar leaves can let existing AD rules remove absent tangents. This experiment uses JAX's rules rather than implementing a separate symbolic derivative rule for every operation. Scaly can use the same principle at its own representation boundary: expose the dependency information to the existing AD/simplifier before adding a new full AD algorithm.
16. The order of DCE and second differentiation matters. Tracing a complete gradient again for every Hessian row duplicated enormous graphs. Prune the needed gradient subgraph first, then differentiate that smaller graph. The local subgraph attempt reduced the intermediate representation relative to naive scalar tracing, though its M=9 native compilation still proved expensive.
17. Bound construction memory and compilation time as well as evaluation time. Whole-program chain Hessian expansion exceeded 13 GB of resident memory and was stopped. Naive local scalar M=9 construction also had to be stopped. The fresh Scaly M=17 Hessian build took about 235 seconds in this experiment and emitted a roughly 5 MB C source. Repeated kernels and compact loops can matter as much for developer usability as for runtime.
18. Vectorization across repetitions can beat fewer scalar operations. Several scalar-local aotax candidates run close to or faster than Scaly despite higher XLA operation estimates. Investigate Scaly loop schedules that make independent stage/car instances contiguous and vectorizable. Test this against current scalarization, loop fusion and range-widening policies. It is a hypothesis for Scaly, not a measured Scaly improvement.
19. Do not apply scalarization or vector widening globally. Full-stage dense derivatives lose on larger chain problems; scalar expansion can explode; neural products prefer matrix operations. The right boundary depends on local support, graph size and operation mix. Expose and record the chosen boundary.
20. Compiler-level operation counts are diagnostic, not a native cost oracle. XLA flops and a count of Scaly's lowered arithmetic nodes are different models. Neither measures final machine instructions, common-subexpression removal, register spills, cache behavior or vector instructions reliably. The scalar-local results demonstrate the mismatch. Keep the count, but judge a change with native interleaved timing.
21. Keep transcendental counts separate. A floating multiply and a scalar libm call have very different cost. Fused broadcasts can also repeat expensive calls. Count and inspect transcendental evaluation before and after batching/fusion; do not hide it inside one total operation count.
22. Test the actual backend boundary. Small tensor contractions in the first composition attempt made IREE emit initialized integer globals unsupported by aotax's host. Explicit tiny sums avoided the problem. Scaly's own lowerer and C renderer likewise need checks of the actual lowered representation, not only mathematically equivalent high-level expressions.
23. Exactness tests must check omitted entries. A values-only comparison cannot detect a missing structural entry. The existing harness checks every lower-triangle or Jacobian entry, including zeros not stored. Retain that rule when adding a new sparse strategy or changing recovery order.
24. Preserve custom derivative semantics and the nonfinite contract. Inlining a primal is not equivalent to honoring an arbitrary custom JVP/VJP. Dropping zero products or reassociating sums can also change NaN and signed-zero behavior. Prove the applicable rule or fall back. The benchmark's custom tanh rule is a special, verified case, not a general permission.
25. A scan is not automatically a sparse sequence of stages. A nonlinear carried state can couple every later stage to earlier controls. Distinguish local multiple-shooting factors from single-shooting rollouts. Dense derivatives or fallback may be the mathematically correct result.
26. A nonlinear reduction can destroy partial separability. The Hessian of a squared sum of local nonlinear terms has cross-factor contributions. `generic_checks.py` deliberately checks that the automatic prototype returns a dense pattern for this case. Do not infer Hessian blocks solely from the map before a reduction.
27. Keep original baselines and forced methods available. A new automatic method should report its local sizes, retained shared maps, derivative method, temporary sizes and fallback reasons. A benchmark column that silently changes its underlying algorithm makes regressions and attribution harder to diagnose.
28. Preserve a usable path while extending coverage. A proved local path plus a clearly reported whole-function fallback is more useful than an unfinished general elimination framework. Subgraph AD, partial separability, local coloring and shared-map composition can be introduced independently behind one stable values/coordinates interface.

29. A Hessian need not require a separate primal separability recognizer. Small output dependency neighborhoods in the gradient give an alternative entry point through sparse Jacobian extraction. Scaly already differentiates its gradient for global Hessian coloring; a structural extraction pass at that boundary could reuse its Jacobian machinery. Compare it with direct factor-Hessian composition and retain shared gradient work. This is a proposed Scaly experiment.

30. Separate algorithmic and compiler explanations when attributing a speedup. Scaly's inspected Hessian path still uses global star coloring, while aotax's scalar-local candidates can be fast despite a larger XLA arithmetic estimate. The useful distinction is the work that survives specialization, simplification and lowering, together with its schedule. A label such as "global" or "structured" is not enough to explain the result.
