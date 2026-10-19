# CPU benchmarks

Run all discovered workloads, or select names and sizes:

```sh
uv run --group bench benchmarks/bench.py
uv run --group bench benchmarks/bench.py --workloads npmpc,npmpc_jac --sizes 12,50 --reps 3 --min-time 0.5s
```

Each binary checks its output against an independent dense reference before timing.
Hessians check the lower triangle; Jacobians check every matrix entry. Stored values
must agree within `1e-9 + 1e-9 * abs(reference)`, and missing entries must be zero.
The `scaly` column uses Scaly's symbolic compiler. `aotax` uses `aotax.sparse` with
`method="asdex"`; `aotax_structured` uses `method="auto"`. Each aotax library has a
`sparse-report.txt` with its extraction/fallback report and Python construction time.
Reports name the local derivative method, repetition counts, input widths and index maps, or
the operation that required fallback. Local Hessians come from weighted primal sums;
Jacobian outputs sharing nonlinear work use one local program per instance. Shared matrix
parameters remain unbatched. Small non-matrix programs use scalar leaves; local Jacobians
choose forward mode when input width is at most output count, and reverse mode otherwise.
The static selection budget is 128 inputs and 512 array operations per local program.
Wider non-matrix Hessians group terms with identical input neighborhoods, with input width
times the sizes of local arguments, results and operands limited to 262,144. There is no runtime
tuning or workload-specific selection.
Construction time excludes compilation and native timing.
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
- `function(size: int) -> Callable` returns the plain JAX scalar Lagrangian or array
  constraint function, with the same positional inputs. The harness constructs both
  sparse derivatives with respect to argument zero.
- `lower_triangle: bool = True` selects Hessian checks. Set it to `False` for Jacobians.

For example, after defining the four functions in `workloads/my_problem.py`:

```python
from workload import Workload

WORKLOADS = (
  Workload("my_problem", (12, 50), inputs, reference, scaly_kernel, lagrangian),
  Workload("my_problem_jac", (12, 50), jac_inputs, jac_reference, scaly_jac, constraints, lower_triangle=False),
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

The programs compare native C with JAX's Accelerated Linear Algebra (XLA) CPU compiler.
Modules export `ProgramWorkload` from `workload.py` alongside the sparse `Workload`.
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
`build/<name>/N<size>/aotax/metadata.json` records size axes, output shapes and integer results
for base and perturbed inputs. Check that counts agree when comparing convergence-dependent rows;
they agree in all default Newton cases.

| Program | Size axes and defaults | Timed work |
|---|---|---|
| `chain_rollout` | M x N: 3x12, 3x50, 3x200, 5x50, 10x50 | Fourth-order Runge-Kutta (RK4) scan; final state and full trajectory |
| `newton` | n: 4, 8, 16, 32, 64 | Bratu Newton, residual infinity norm <= 1e-10, at most 20 iterations; solution, count and residual |
| `gauss_newton` | observations x parameters x iterations: 16x3x8, 64x3x8, 64x5x12 | Exponential mixture fit, damped normal equations, Cholesky and triangular solves; parameters, residual and fixed count |
| `chain_qp` | M x N x iterations: 3x4x15, 3x12x15, 5x12x15 | Condensed chain quadratic program (QP), fixed-step primal-dual method; primal, slacks, multipliers, residuals and count |
| `batched_newton` | batch x n: 1x8, 4x8, 16x8, 64x8 | vmap of converging Newton; per-lane solutions, counts and residuals |

The rollout and QP use `_chain.initial_state(M)` and `_chain.PARAMS`, including the
canonical dt=0.2. Controls start at zero. QP construction differentiates the rollout
and uses Scaly's chain weight constants and [-1, 1] control bounds. Stage costs divide
by the selected horizon N; `_chain.stage_cost_jax` uses its fixed horizon of 40. Construction
is outside timing; Q, c, G and h are runtime inputs. This is one dense condensed
subproblem, not a complete nonlinear controller or a scalable stagewise solver.
The fit has an odd parameter count, with amplitude/rate pairs and one offset. The
3-parameter case is the H0 model. Batched Newton uses Bratu parameters from 0.5 to 2.0;
its loop runs until the slowest lane finishes and masks converged lanes.
