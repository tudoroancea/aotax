# Sparse kernel benchmarks

Run all discovered workloads, or select names and sizes:

```sh
uv run --group bench benchmarks/bench.py
uv run --group bench benchmarks/bench.py --workloads npmpc,npmpc_jac --sizes 12,50 --reps 3 --min-time 0.5s
```

Each binary checks its output against an independent dense reference before timing.
Hessians check the lower triangle; Jacobians check every matrix entry. Stored values
must agree within `1e-9 + 1e-9 * abs(reference)`, and missing entries must be zero.
Results use the best CPU time across interleaved fresh processes. Generated files live
in `build/<workload>/N<size>/<backend>/`; `build/results.csv` includes every repetition.

## Add a workload

Add a module in `workloads/` exporting `WORKLOADS`, a sequence of `workload.Workload`.
Discovery imports every Python module whose name does not start with `_`. No registry
or command-line edits are needed. Put helpers in a file starting with `_`.

`Workload` has these fields:

- `name: str` and `sizes: tuple[int, ...]` give the command-line name and default sizes.
- `inputs(size: int) -> dict[str, np.ndarray]` gives named float64 arrays. Insertion
  order must match both kernels' positional argument order. Scalars use shape `()`.
- `reference(size: int, inputs: dict[str, np.ndarray]) -> np.ndarray` returns a dense
  two-dimensional matrix, independent of asdex's sparsity detection.
- `scaly(size: int) -> tuple[sc.Function, np.ndarray, np.ndarray]` returns a function
  with one sparse output, its row indices and its column indices.
- `aotax(size: int) -> tuple[Callable, np.ndarray, np.ndarray]` returns a JAX function
  of the same inputs, its sparse output rows and columns. The function returns only
  the one-dimensional float64 values array, in those coordinates' order.
- `lower_triangle: bool = True` selects Hessian checks. Set it to `False` for Jacobians.

For example, after defining the four functions in `workloads/my_problem.py`:

```python
from workload import Workload

WORKLOADS = (
  Workload("my_problem", (12, 50), inputs, reference, scaly_kernel, aotax_kernel),
  Workload("my_problem_jac", (12, 50), jac_inputs, jac_reference, scaly_jac, aotax_jac, lower_triangle=False),
)
```

Match Scaly's `benchmarks/harness/sweep.py` kernel definitions and synthetic input
sampling. Use the solver descriptor's full constraint vector, including inequalities.
For Hessians, use the exact Lagrangian and select its lower triangle. A scalar objective
multiplier may be fixed to one if documented. Build large dense references by scattering
plain JAX stage derivatives when whole-problem differentiation is too expensive.

The npmpc workloads retain the original trained 32-wide decoder, horizon samples and
mixed-sign constraint multipliers. Its objective multiplier is fixed to one. They use
synthetic samples rather than Scaly's optional harvested closed-loop inputs. The
npmpc_decoder workloads instead sweep the decoder's width at horizon 12, with Scaly's
seed-0 untrained weights.

## Complete programs

```sh
uv run --group bench benchmarks/bench.py --workloads chain_rollout,newton,gauss_newton,chain_qp,batched_newton --reps 3
uv run --group bench benchmarks/bench.py --workloads chain_rollout --sizes 3x12,5x50
```

These modules export `ProgramWorkload` from `workload.py` alongside the sparse `Workload`.
Its fields are `name`, named `sizes`, `axes(size) -> dict[str, int]`,
`inputs(size) -> tuple` of positional arguments, `function(size) -> Callable`, and
`tolerances(size) -> tuple[(rtol, atol), ...]` in JAX pytree leaf order. Arguments and
results can be pytrees. Leaves are dense float64, int32 or int64 arrays, including scalars.
A CPU `jax.jit` invocation supplies the reference. Every float output has its own
absolute and relative tolerance; integer leaves always require exact equality.
Nonfinite references fail. The build also checks a second, slightly perturbed input.
Each native binary reads the base reference and checks every leaf before timing, so a
bad output or iteration count aborts the benchmark. Its buffers are 64-byte aligned.

The program table has three columns:

- `aotax`: CPU time inside the Google Benchmark C++ binary, calling the C API directly.
  Context creation, compilation, input loading and correctness checks are outside timing.
- `XLA loop`: elapsed time per evaluation inside one CPU-jitted `lax.fori_loop`, divided
  by its evaluation count. Every floating input leaf is scaled by
  `1 + 1e-8 * (iteration % 2)`. Each evaluation contributes the sum of every output
  leaf to a checksum. This prevents constant-input hoisting and discarding earlier
  evaluations. Input scaling, output reduction and loop overhead are included.
  Constant subexpressions and invariant integer inputs may still be hoisted by XLA.
- `JAX Python`: elapsed time for a CPU-jitted call from Python, with device inputs and
  `block_until_ready` for every call. It includes Python dispatch and synchronization.

Compilation and warmup are excluded. The XLA loop calibrates its count toward
`--min-time`, with 32 to 100,000 evaluations. Each repetition runs the native binary
and then a fresh Python process for the two JAX measurements. The best of each is
reported, and `build/results.csv` retains all repetitions with backend names `aotax`,
`xla`, `jax_python`. Sparse workloads retain their existing Scaly comparison and CSV rows.
`build/<name>/N<size>/aotax/metadata.json` records size axes, output shapes and integer results.

| Program | Size axes and defaults | Timed work |
|---|---|---|
| `chain_rollout` | M x N: 3x12, 3x50, 3x200, 5x50, 10x50 | RK4 scan; final state and full trajectory |
| `newton` | n: 4, 8, 16, 32, 64 | Bratu Newton, residual infinity norm <= 1e-10, at most 20 iterations; solution, count and residual |
| `gauss_newton` | observations x parameters x iterations: 16x3x8, 64x3x8, 64x5x12 | Exponential mixture fit, damped normal equations, Cholesky and triangular solves; parameters, residual and fixed count |
| `chain_qp` | M x N x iterations: 3x4x15, 3x12x15, 5x12x15 | Condensed chain QP, fixed-step primal-dual method; primal, slacks, multipliers, residuals and count |
| `batched_newton` | batch x n: 1x8, 4x8, 16x8, 64x8 | vmap of converging Newton; per-lane solutions, counts and residuals |

The rollout and QP use `_chain.initial_state(M)` and `_chain.PARAMS`, including the
canonical dt=0.2. Controls start at zero. QP construction differentiates the rollout
and uses Scaly's chain stage/terminal weights and [-1, 1] control bounds. Construction
is outside timing; Q, c, G and h are runtime inputs. This is one dense condensed
subproblem, not a complete nonlinear controller or a scalable stagewise solver.
The fit has an odd parameter count, with amplitude/rate pairs and one offset. The
3-parameter case is the H0 model. Batched Newton uses Bratu parameters from 0.5 to 2.0;
its loop runs until the slowest lane finishes and masks converged lanes.

## Scaly devrush solver study

```sh
SCALY_DEVRUSH_CHECKOUT=/Users/tudoroancea/dev/scaly \
  uv run --group bench benchmarks/scaly_devrush.py --reps 3
```

The checkout must have `origin/devrush` locally. The script resolves its commit and
reads `src/scaly` with `git archive` into a temporary directory. It never switches the
checkout, installs that source or changes the project's Scaly main dependency.
Solver generation runs in a separate process with that source first on `sys.path`.
Generated C uses `-O3 -mcpu=native -fno-math-errno`. Native calls run in a C timing loop;
creation and compilation are excluded. Results and repetitions go to `build/devrush.json`.
Use `--newton-sizes 8` and `--qp-sizes 3x4x15` for a smoke run.

Newton uses the same residual, zero initial guess, dense LU, tolerance and iteration
limit; the study requires equal iteration counts as well as matching solutions and
successful Scaly status. The chain QP uses the same Q and c and control bounds, but
Scaly's dense proximal interior-point solver has its own initialization, scaling,
predictor/corrector steps and stopping rules, with at most 30 iterations and absolute
and relative tolerances of 1e-9. Its iteration count and primal residual are reported.
Scaly embeds Q at generation time; the JAX program accepts Q at runtime. This is a
complete-solver comparison with different algorithms and specialization, not a direct
compiler comparison. Both solutions must agree within `2e-6 + 1e-5 * abs(reference)`.
