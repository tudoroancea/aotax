# aotax

Compile a JAX function ahead of time into a static C library and a header. The program that links
it needs no Python, no JAX and no just-in-time compiler: only the archive and libm.

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
cc main.c build/hess/libhess.a -lm   # on Linux, also -lpthread -ldl
```

`{name}_call` takes the function's inputs and then its outputs, flattened in JAX's pytree order, as
pointers to contiguous row-major arrays of `float`, `double`, `int32_t` or `int64_t`. The header
lists their shapes.

## How it works

aotax is a thin layer over [IREE](https://iree.dev). It exports the function's StableHLO with
`jax.export`, and `iree-compile` turns that into two pieces: the dispatches' machine code as an
object file, and the program that schedules them, IREE's virtual machine (VM) module, written out
as C. aotax generates the C host that runs that module on IREE's single-threaded `local-sync`
device behind the three functions above. It then merges the host, the dispatches and IREE's runtime
into one archive.

Three compiler flags matter, none of them for speed:

- `--iree-input-demote-f64-to-f32=false` (and the same for i64), or IREE silently computes in f32;
- `--iree-llvmcpu-link-embedded=false` with `--iree-llvmcpu-link-static`, because IREE's default
  self-contained ELF has no f64 `sin`, `cos` or `exp`; the dispatches become a plain object whose
  calls resolve against libm;
- `--output-format=vm-c`, so the module comes out as C rather than as bytecode for the runtime to
  interpret.

The PyPI runtime wheel has IREE's runtime sealed inside its Python extension, so on first use aotax
builds the C runtime from source, at the installed compiler's release and with only the
`local-sync` driver. This takes about a minute and needs git and a C compiler; CMake and Ninja come
from PyPI. The result is cached in `~/.cache/aotax` (or `$AOTAX_CACHE`).

## Limits

- Code is compiled for the host CPU; there is no cross-compilation.
- Shapes are static, and inputs and outputs are dense `float32`, `float64`, `int32` or `int64`.
- Each call copies the inputs into buffers allocated once per context, and copies each result,
  which the runtime allocates, out to caller memory.
- The runtime adds about 700 KiB to a program.
- Tested on macOS on Apple silicon; the Linux path follows the same experiment, which ran on
  x86-64, but has no tests yet.

## Benchmarks

`benchmarks/` compiles the lower triangle of a Lagrangian Hessian from a neural-network model
predictive control problem, taken from [Scaly](https://github.com/PREDICT-EPFL/scaly), twice: by
Scaly's own symbolic compiler, and from the same Lagrangian in JAX, differentiated sparsely by
[asdex](https://adrianhill.de/asdex/) and compiled by aotax. Both are linked into one Google
Benchmark harness, which checks every entry against a dense `jax.hessian` before timing.

```bash
uv run --group bench benchmarks/bench.py
```

CPU time per evaluation on an Apple M4 Max, best of three processes:

| horizon N | Scaly | aotax | ratio |
|---:|---:|---:|---:|
| 12 | 14.7 µs | 30.7 µs | 2.09× |
| 50 | 60.3 µs | 92.2 µs | 1.53× |
| 200 | 242.7 µs | 322.9 µs | 1.33× |

aotax does not try to compete with Scaly here. The comparison shows what a general JAX function
costs when it is compiled to C with no tuning.

## License

BSD 2-Clause.
