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
synthetic samples rather than Scaly's optional harvested closed-loop inputs. Decoder
width sweeps are not included.
