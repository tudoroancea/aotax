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

`aotax.compile(..., device="metal")` uses the same HAL analysis and control-flow translation to
generate an Objective-C host for Metal 4. It embeds IREE's Metal Shading Language (MSL) source and
compiles cached pipelines at `create`, without Xcode's optional Metal compiler. Each context owns
shared buffers, an argument table per dispatch site, a residency set, and a reusable queue and
command-buffer object. Dispatches, copies and fills use one compute encoder per queue execution.
HAL barriers preserve dependencies; fences and host reads block on a shared event. Inputs and
outputs always use staging copies. The generated header lists `-framework Metal -framework Foundation`,
and `Library.load()` works as on the CPU.

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

On the CPU, `{name}_create` allocates aligned storage for transients, loop-carried buffers, and workgroup memory.
Constants are embedded as aligned static data; any constant preparation runs at creation. The host
allocates no heap memory during calls; Accelerate may allocate internally.
It preserves subnormal values by default. With `approximate_math=True`, it flushes them to zero
as IREE and XLA's CPU backend do. Both modes restore the caller's floating-point mode afterward. Inputs bind directly
to the caller's arrays when the HAL program never writes them. Outputs bind directly when the HAL program gives them
a separate buffer. Results that share storage with inputs, constants, or
other results need a copy.

Pointers need no special alignment. A read-only binding aligned to 64 bytes avoids staging; aotax copies
less-aligned arrays and inputs whose bindings the HAL program writes through context-owned buffers.
This meets the dispatch code's alignment assumptions and preserves the caller's inputs.
Inputs may overlap outputs, in which case the affected inputs are copied before execution. Outputs
may overlap each other and are written in the order listed in the header.

Elementary functions use libm-quality results by default. On the CPU, `sin`, `cos`, `exp`, `log`
and `tanh` stay in dispatches; their `float32` inputs are promoted to `float64` and their results
converted back. Other elementary functions use generated libm adapters because IREE expands
some of them inaccurately even in `float64`, including `expm1` and `log1p` near zero.
The default covers trig and inverse trig, hyperbolic and inverse hyperbolic functions, exponentials,
logarithms, `power`, `cbrt`, `erf` and `erfc`. Dense sweeps, including large arguments and special
values, stay within 2 units in the last place (ulp) of NumPy or the system libm on the tested machine.
This is a check of individual functions, not an error bound for a whole compiled program.

Use `aotax.compile(..., approximate_math=True)` to allow faster approximations. On the CPU this
enables aotax's vectorized `float64` `exp` polynomial and IREE's built-in approximations. The
`exp` polynomial stays within 2 ulp after flushing subnormal results to zero, and handles NaN
and infinities. IREE's approximations have no common error bound. Its
[math transform pass](https://github.com/iree-org/iree/blob/v3.12.0/compiler/src/iree/compiler/Codegen/Common/MathTransform.cpp)
has no option to disable them in 3.12, which is why the accurate path uses promotion and adapters.
CPU `float64` `cbrt` uses libm in both modes because IREE cannot lower it.

On Metal, the default runs elementary functions through host libm adapters, waiting for preceding
GPU work and writing results into shared buffers. This adds synchronization: npmpc Hessians run
3 to 5 times slower than with opt-in, in single calls and in batches of 128. Opting in keeps these operations on the GPU with IREE's approximations.
Both modes use `MTLMathModeSafe`; the default also selects `MTLMathFloatingPointFunctionsPrecise`,
and opt-in selects `Fast`. These are separate options. Safe alone leaves native math functions at
Fast and cannot repair polynomials already emitted by IREE. See Apple's
[compile-option semantics](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode).

With `math_before_broadcast=True`, aotax puts compiler barriers after CPU `float64` `sin`, `cos`
and `tanh` results that feed broadcasts. IREE then computes these libm results before broadcasting
them, instead of repeating the calls for the added dimensions. Results are unchanged. It makes the
asdex race-car Jacobians 14 to 19% faster and leaves the structured ones unchanged. The barriers
also block fusion, which slowed unbumpercars at 8 cars by 7% in an earlier measurement.

Four compiler settings matter:

- `--iree-input-demote-f64-to-f32=false` and the same for i64 prevent silent conversion to 32 bits;
- `--iree-llvmcpu-link-embedded=false` with `--iree-llvmcpu-link-static` produces a plain dispatch
  object whose remaining `sin` and `cos` calls resolve against libm, which provides their f64 versions;
- `--iree-llvmcpu-disable-distribution`, because the host runs every dispatch on one thread;
- `--iree-llvmcpu-loop-unrolling`, which made the benchmark below 5 to 7% faster.

## Foreign functions

Declare user C targets explicitly when compiling a JAX foreign function interface (FFI) call:

```python
def twice(x):
  return jax.ffi.ffi_call("twice", jax.ShapeDtypeStruct(x.shape, x.dtype))(x)


lib = aotax.compile(twice, np.ones(8), out="build/twice", foreign={"twice": "my_twice"})
twice_loaded = lib.load(extra=["my_twice.o"])
```

The generated `twice.h` declares the C symbol and the descriptor types. Include it in `my_twice.c`
so the compiler checks the implementation's signature:

```c
#include "twice.h"

int my_twice(const aotax_ffi_tensor* inputs, size_t ninputs,
             const aotax_ffi_tensor* results, size_t nresults, void* scratch) {
  size_t n = 1;
  for (size_t i = 0; i < inputs[0].rank; ++i) n *= inputs[0].shape[i];
  const double* x = inputs[0].data;
  double* y = results[0].data;
  for (size_t i = 0; i < n; ++i) y[i] = 2 * x[i];
  return 0;
}
```

Each descriptor has `void* data`, `aotax_ffi_dtype dtype`, `size_t rank` and
`const int64_t* shape`. The dtype is `AOTAX_FFI_F32`, `AOTAX_FFI_F64`, `AOTAX_FFI_I32` or
`AOTAX_FFI_I64`. Shapes are static, arrays are dense and row-major, and scalars have rank zero
and a null shape pointer. One C symbol can handle several shapes and signatures; the header
lists every specialization's inputs, results and scratch size.

Calls are synchronous. Inputs are borrowed and immutable; fill every result in its fresh buffer
and retain no pointers after returning. Return zero on success or nonzero for an execution error.
Keep numerical status, such as solver convergence, in tensor results. Foreign code runs with the
library's floating-point mode, which preserves subnormals unless `approximate_math=True`.
For temporary storage,
use `aotax.Foreign("my_twice", scratch=256)` in the registry. Scratch lives in the context,
is aligned to 64 bytes and is reused without clearing. With no scratch request the pointer is null.

The archive leaves user symbols unresolved. Compile the implementation, then link it after the archive:

```sh
cc -O2 -fPIC -I build/twice -c my_twice.c -o my_twice.o
cc main.c -I build/twice build/twice/libtwice.a my_twice.o -lm -framework Accelerate
```

Omit `-framework Accelerate` outside macOS. `load(extra=[...])` accepts the same objects, libraries
and link flags after the archive. An XLA-registered handler has a different call-frame ABI;
aotax never calls it. Only explicitly registered user targets receive JAX export exemptions, and no XLA
registration is needed to compile them. CPU tensor calls work inside JAX loops. Backend attributes,
opaque state, effects, tokens, non-row-major layouts and input/output alias requests raise compilation
errors. Pass attribute values as tensor operands instead.

IREE 3.12 can crash on loops carrying several scalar `float64` foreign results. Returning solver
statistics in a vector avoids this compiler failure.

## Limits

- CPU code is compiled for the host CPU; there is no cross-compilation.
- Shapes are static, and inputs and outputs are dense `float32`, `float64`, `int32` or `int64`.
- With `approximate_math=True`, IREE 3.12's float32 `sin`/`cos` on CPU and Metal lose range-reduction
  accuracy around `|x| = 1e3`, with absolute error about `3e-5`, and can return enormous values at
  `1e10`. Subnormals are flushed, small `float64` `expm1`/`log1p` lose accuracy, and powers with
  negative bases or nonfinite operands can differ from libm. Validate the opt-in mode on your inputs.
- The flag controls elementary functions and `erf`/`erfc`. Other JAX special-function algorithms
  and polynomials written directly by the caller retain their own accuracy limits.
- Host dispatches run on one thread; Accelerate controls threading within BLAS calls. A context
  cannot be called concurrently; separate contexts can.
- Metal requires a Metal 4 device and macOS 26 or later, and a macOS 26 or later SDK to build the
  host. Inputs and outputs must be `float32` or `int32`. Float64 intermediates raise an error;
  IREE's internal integer indices keep their original width. Metal does not use Accelerate or
  support user or LAPACK custom calls. Its generated elementary-function adapters run on the host.
  Host control flow is preserved, but each loop iteration that submits work
  pays GPU submission and completion latency. Single small evaluations are much slower than the
  CPU; batching can amortize that cost. A reported GPU error permanently fails the context;
  destroy it and create a new one. See [Metal measurements](notes/metal.md).
- The translator follows the pinned IREE compiler's executable application binary interface (ABI).
  Unsupported HAL operations or command-buffer patterns raise an error during compilation.
- Custom calls support JAX's CPU `lapack_{s,d}getrf_ffi`, `lapack_{s,d}potrf_ffi` and
  `lapack_{s,d}trsm_ffi` targets on macOS. These cover ordinary `jnp.linalg.solve`, Cholesky,
  triangular solves and SciPy-style LU and Cholesky solves. Explicitly declared user targets use
  the C convention above. Undeclared targets and LAPACK calls on other platforms raise an error
  during compilation. LAPACK adapters require static, nonzero matrix and batch dimensions.
- Tested on macOS on Apple silicon. The Linux archive/link path is present but untested.

## Sparse derivatives

Install `aotax[sparse]` for sparse derivatives of plain JAX functions:

```python
from aotax.sparse import sparse_hessian, sparse_jacobian

h = sparse_hessian(energy, x, parameters)
lib = aotax.compile(h.values, x, parameters, out="build/hessian")
```

Both constructors take arrays or `jax.ShapeDtypeStruct`s and differentiate one real floating
array selected by `argnums=0`. Other arguments remain runtime inputs. Hessians require a scalar
output and return its lower triangle. Jacobians flatten one array output and the selected input
in row-major order. The frozen result contains `values`, immutable sorted unique `rows` and `cols`,
`shape`, and a readable `report`. Structural entries remain present when runtime values are zero.
The contract covers finite derivatives and aotax's documented floating-point behavior.
At masked nonsmooth points, derivatives follow JAX's reverse-mode rules and may be NaN
where `jax.jacfwd` gives a finite value.

`method="asdex"` uses structural detection and coloring with direct sparse recovery.
`method="auto"` extracts local programs from the primal function and otherwise uses asdex,
with the reason in the report. The asdex path rejects statically out-of-bounds gather and
scatter indices because asdex 0.5.2 can omit entries for these operations. Structured
extraction supports clipped and filled gathers; out-of-bounds `promise_in_bounds` gathers
raise because they violate the indexing promise. Unsupported asdex operations raise a
`ValueError` naming the operation. Index checks also inspect nested control flow and custom
derivative programs. While-loop Jacobians use forward mode when reverse mode is unavailable.
Outputs that share nonlinear work are grouped before
Jacobian differentiation. Hessians differentiate weighted local primal sums with forward
mode over reverse mode, then add their overlapping lower triangles. Nonlinear functions of
sums remain intact. Duplicate input coordinates are deduplicated before differentiation.
When static gathers reuse a computed map, extraction retains its live dependent outputs,
evaluates the map and its derivatives once per instance, and composes local Jacobians and
Hessians with downstream adjoints. Overlapping slices can use the same composition when
flat extraction exceeds its budget. Input-independent outputs remain runtime coefficients.

Small programs without matrix products use scalar input leaves to remove absent tangents.
Wider Hessians group terms with the same input neighborhood, including repeated neighbor
coupling inside a stage, while keeping different neighborhoods separate.
Jacobians use forward mode when the local input width is at most the output count, and
reverse mode otherwise. Matrix products retain their contractions; shared parameters use
`vmap` with `in_axes=None`. Pointwise custom Jacobian-vector product (JVP) rules are preserved.
`method="structured"` raises when extraction exceeds its supported rules or construction budget.

Selection uses graph properties, without timing kernels. Each extraction pass has at most 256 local
programs, each with at most 128 selected inputs, 512 array operations and a backward-demand depth of 128.
Before scalar Hessian differentiation, input width times the sizes of local arguments, results and
operands, with custom JVP rule bodies counted two levels deep, must not exceed 262,144.
Groups exceeding the array-operation budget split into smaller sums. Wide Jacobian
neighborhoods stay separate; small outputs sharing nonlinear work are merged. Shared nonlinear
maps with multiple computed gather sources or a different floating dtype from the selected input,
general control flow and non-pointwise custom rules also fall back.
Shared-map derivatives and their composition each have a two-million-entry construction budget.

## Benchmarks

`benchmarks/` times two kinds of workloads on the CPU; `benchmarks/README.md` describes them and
how to add one.

```bash
uv run --group bench benchmarks/bench.py
```

The derivative workloads come from [Scaly](https://github.com/PREDICT-EPFL/scaly)'s benchmark
suite: sparse Lagrangian Hessians and constraint Jacobians for chain and race-car control,
a neural-process controller with horizon and decoder-width sweeps, and a learned-car safety filter.
Scaly compiles them with its own symbolic compiler; aotax compiles the
same functions written in JAX through `aotax.sparse`, in two ways: classical
[asdex](https://adrianhill.de/asdex/) coloring (`method="asdex"`, the `aotax` column of
`bench.py`) and structured extraction (`method="auto"`, `aotax_structured`), which takes the
structured path on every workload here. Each library has a `sparse-report.txt` recording its
method and construction time. Every entry is checked against a dense JAX derivative before timing.
aotax CPU time divided by Scaly's on an Apple M4 Max, best of five fresh processes, with the default
accurate math:

| Problem | Sizes | Hessian, asdex | Hessian, structured | Jacobian, asdex | Jacobian, structured |
|---|---|---|---|---|---|
| npmpc, neural-network dynamics | horizon 12, 50, 200 | 0.98×, 0.88×, 0.89× | 0.74×, 0.64×, 0.61× | 0.90×, 0.81×, 0.75× | 0.71×, 0.63×, 0.50× |
| npmpc_decoder, network width | width 16, 64, 256 | 1.59×, 0.41×, 0.16× | 1.10×, 0.33×, 0.12× | 1.06×, 0.55×, 0.17× | 0.73×, 0.45×, 0.14× |
| unbumpercars, network plant | 2, 4, 8 cars | 0.23×, 0.23×, 0.42× | 0.17×, 0.11×, 0.08× | 0.44×, 0.34×, 0.39× | 0.40×, 0.23×, 0.17× |
| chain of masses | 3, 9, 17 masses | 3.46×, 3.02×, 2.01× | 0.70×, 2.36×, 1.67× | 16.6×, 5.42×, 4.26× | 1.43×, 1.47×, 0.51× |
| race cars | horizon 5, 40, 500 | 4.43×, 3.28×, 3.95× | 1.80×, 1.23×, 1.44× | 5.63×, 5.01×, 4.99× | 1.41×, 1.07×, 1.05× |

aotax is faster where neural networks dominate: their matrix products run on Apple's matrix
coprocessor through Accelerate, and the rest vectorizes. With asdex it is slower on small dynamics
without matrix products. There most of the gap is the derivative algorithm rather than the compiler:
asdex's colored products do 4.5 to 11.5 times the arithmetic of Scaly's symbolic derivatives, and
XLA running the same asdex functions is also 5 to 9 times slower than Scaly. Structured extraction
removes most of that gap: race-car Jacobians come within 7% of Scaly at horizons 40 and 500, and the
chain Jacobian at 17 masses takes half of Scaly's time. The chain Hessian at 9 and 17 masses
remains 1.7 to 2.4 times slower.

The accurate default costs little on these workloads. In an interleaved comparison, best of five
fresh processes per mode, opt-in makes npmpc 2 to 6% faster and leaves race cars unchanged. On
65,536 float32 elements, `sin(x)*x + 2` is 3.6 times faster with opt-in, because the default
computes `sin` in float64.

The complete programs run whole algorithms written in JAX with loops: a fourth-order Runge-Kutta
rollout, a Newton solve with `while_loop` and LU factorizations, a Gauss-Newton fit with Cholesky
solves, a 15-step interior-point method for a quadratic program, and a batched Newton solve. aotax
is compared with XLA running the same function inside one jitted loop, which excludes Python's
call overhead. aotax time divided by XLA's, from the same run:

| Program | Sizes | aotax / XLA |
|---|---|---|
| rollout | 3×12, 3×50, 3×200, 5×50, 10×50 | 0.42×, 0.54×, 0.85×, 0.67×, 0.95× |
| Newton | n = 4, 8, 16, 32, 64 | 0.83×, 1.14×, 1.25×, 1.43×, 1.25× |
| Gauss-Newton | 16×3×8, 64×3×8, 64×5×12 | 0.48×, 0.50×, 0.40× |
| quadratic program | 3×4×15, 3×12×15, 5×12×15 | 0.66×, 0.90×, 0.91× |
| batched Newton | batch 1, 4, 16, 64 | 1.24×, 0.96×, 1.08×, 1.07× |

Newton loses at larger n because hundreds of small dispatches surround each factorization.

With `approximate_math=True` on the GPU, `device="metal"` pays off only for batches: from about
32 to 48 npmpc Hessians at horizon 12, or 10 to 12 at horizon 50, it beats the CPU, by 4.5× and
6.3× at 128. A single call costs about 400 µs against 11 µs on the CPU. With the accurate default,
the GPU beats the CPU at 128 Hessians by only 1.1× at horizon 12 and 2.2× at horizon 50. `notes/metal.md` has the measurements.

The first version of aotax ran IREE's own runtime and took 30.7, 92.2 and 322.9 µs on the npmpc
Hessian, reduced to 15.9, 58.3 and 200.9 µs with the approximations now made opt-in.

## License

BSD 2-Clause.
