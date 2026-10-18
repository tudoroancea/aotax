# aotax

> [!WARNING]
> aotax is an experiment, not a finished library. Its interface, generated code and supported
> features change without notice, and it has only been tested on macOS on Apple silicon. Do not
> use it in production or in a project that depends on it.

Compile a JAX function ahead of time into a static C library and a header. The program that links
it needs no Python, no JAX and no just-in-time compiler: only the archive, libm and, on macOS, Accelerate.

```python
import jax
import jax.numpy as jnp
import numpy as np
import aotax

jax.config.update("jax_enable_x64", True)


def energy(q):
  return jnp.sum(jnp.cos(q[1:] - q[:-1])) + 0.5 * q @ q


lib = aotax.compile(jax.hessian(energy), np.zeros(8), out="build/hess", name="hess")
hess = lib.load()  # the same library, called from Python, to check it against JAX
np.testing.assert_allclose(hess(np.ones(8)), jax.hessian(energy)(np.ones(8)))
```

`build/hess/` now holds `hess.h` and `libhess.a`:

```c
#include "hess.h"

hess_context* context;
if (hess_create(&context)) return 1;
double q[8] = {0}, H[8 * 8];
if (hess_call(context, q, H)) return 1;
hess_destroy(context);
```

```bash
cc main.c build/hess/libhess.a -lm
```

On macOS, also pass `-framework Accelerate`. The generated header lists the required link flags.

`{name}_call` takes the function's inputs and then its outputs, flattened in JAX's pytree order, as
pointers to contiguous row-major arrays of `float`, `double`, `int32_t` or `int64_t`. The header
lists their shapes.

## How it works

aotax exports the function with `jax.export` as Stable High Level Operations (StableHLO).
[IREE](https://iree.dev) compiles its dispatches into a machine-code object and writes the host
program in its hardware abstraction layer (HAL) intermediate representation. aotax reads that
program with IREE's compiler bindings and generates C that calls the dispatches directly. Branches
and loops become C control flow; copies and fills run in order when their command buffer executes.
The archive contains only the generated host and dispatch objects. There is no IREE runtime.

Before IREE compilation, supported LAPACK custom calls become private tensor function calls.
IREE preserves their inputs and results in the HAL program, where the generated host calls
static C adapters. On macOS these use Accelerate for LU, Cholesky and triangular solves in
`float32` and `float64`, including batches and calls inside loops. The adapters convert row-major
matrices to LAPACK's column-major layout for factorization, preserve inputs, and keep results
and scratch in the context. LAPACK status stays in the tensor program, so singular solves and
invalid Cholesky factorizations retain JAX's handling of nonfinite results.

On macOS, aotax also reads IREE's dispatch sources from the same compilation. Pure matrix products
call the Basic Linear Algebra Subprograms (BLAS) in Apple's Accelerate framework when they need
at least 32,768 floating-point operations per matrix for `float64`, or 65,536 for `float32`.
The host passes their buffer offsets, strides and transpose flags directly; batched products loop
over BLAS calls. Smaller products, fused operations and layouts the translator cannot map exactly
keep IREE's dispatch code. Other platforms use IREE for all dispatches.

`{name}_create` allocates aligned storage for transients, loop-carried buffers, and workgroup memory.
Constants are embedded as aligned static data; any constant preparation runs at creation. The host
allocates no heap memory during calls. It flushes subnormal values to zero during execution, as IREE does,
and restores the caller's floating-point mode afterward. Inputs bind directly to the caller's arrays
when the HAL program never writes them. Outputs bind directly when the HAL program gives them
a separate buffer. Results that share storage with inputs, constants, or
other results need a copy.

Pointers need no special alignment. A read-only binding aligned to 64 bytes avoids staging; aotax copies
less-aligned arrays and inputs whose bindings the HAL program writes through context-owned buffers.
This meets the dispatch code's alignment assumptions and preserves the caller's inputs.
Inputs may overlap outputs, in which case the affected inputs are copied before execution. Outputs
may overlap each other and are written in the order listed in the header.

Before IREE sees the program, aotax replaces every `float64` `exp` with a range reduction and a
polynomial built from ordinary arithmetic, so that it vectorizes with the rest of its dispatch
instead of calling libm once per element. It stays within 2 units in the last place (ulp) of libm
over the whole range, and handles infinities, NaN and underflow to zero.

Four compiler settings matter:

- `--iree-input-demote-f64-to-f32=false` and the same for i64 prevent silent conversion to 32 bits;
- `--iree-llvmcpu-link-embedded=false` with `--iree-llvmcpu-link-static` produces a plain dispatch
  object whose remaining `sin` and `cos` calls resolve against libm, which provides their f64 versions;
- `--iree-llvmcpu-disable-distribution`, because the host runs every dispatch on one thread;
- `--iree-llvmcpu-loop-unrolling`, which made the benchmark below 5 to 7% faster.

## Limits

- Code is compiled for the host CPU; there is no cross-compilation.
- Shapes are static, and inputs and outputs are dense `float32`, `float64`, `int32` or `int64`.
- Host dispatches run on one thread; Accelerate controls threading within BLAS calls. A context
  cannot be called concurrently; separate contexts can.
- The translator follows the pinned IREE compiler's executable application binary interface (ABI).
  Unsupported HAL operations or command-buffer patterns raise an error during compilation.
- Custom calls support JAX's CPU `lapack_{s,d}getrf_ffi`, `lapack_{s,d}potrf_ffi` and
  `lapack_{s,d}trsm_ffi` targets on macOS. These cover ordinary `jnp.linalg.solve`, Cholesky,
  triangular solves and SciPy-style LU and Cholesky solves. Other targets, user foreign calls,
  and LAPACK calls on other platforms raise an error during compilation. Adapters require static,
  nonzero matrix and batch dimensions; JAX may eliminate empty calls before export.
- Tested on macOS on Apple silicon. The Linux archive/link path is present but untested.

## Benchmarks

`benchmarks/` compiles the lower triangle of a Lagrangian Hessian from a neural-network model
predictive control problem, taken from [Scaly](https://github.com/PREDICT-EPFL/scaly), twice: by
Scaly's own symbolic compiler, and from the same Lagrangian in JAX, differentiated sparsely by
[asdex](https://adrianhill.de/asdex/) and compiled by aotax. Both are linked into one Google
Benchmark harness, which checks every entry against a dense `jax.hessian` before timing.

```bash
uv run --group bench benchmarks/bench.py
```

CPU time per evaluation on an Apple M4 Max, best of five processes:

| horizon N | Scaly | aotax | ratio |
|---:|---:|---:|---:|
| 12 | 15.0 µs | 15.9 µs | 1.06× |
| 50 | 61.3 µs | 58.2 µs | 0.95× |
| 200 | 246.4 µs | 202.1 µs | 0.82× |

The first version of aotax ran IREE's own runtime and took 30.7, 92.2 and 322.9 µs. Generating the
host as C removed about 14 µs of runtime work from every call. Most of the gain at the larger
horizons comes from Accelerate: the matrix products of the network's layers, batched over stages
and Hessian colors, run on Apple's matrix coprocessor. On other platforms they stay in IREE's code.
The largest remaining cost, about a third of the time at every horizon, is in dispatches where
IREE fuses a matrix product with elementwise work and generates scalar code for both.

## License

BSD 2-Clause.
