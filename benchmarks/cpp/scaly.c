// Scaly's generated function behind kernel.h, called through its pointer ABI as Scaly's own
// benchmark cells call it: input pointers, the output written in place, a caller-owned workspace.
#include <stdlib.h>

#include "kernel.h"
#include "npmpc_hess.h"

static const double* g_arg[npmpc_hess_SZ_ARG];
static double* g_res[npmpc_hess_SZ_RES];
static double g_w[npmpc_hess_SZ_W > 0 ? npmpc_hess_SZ_W : 1];

void kernel_setup(const double* z, const double* p, const double* lam_g) {
  g_arg[0] = z;
  g_arg[1] = p;
  g_arg[2] = lam_g;
}

void kernel_evaluate(double* out) {
  g_res[0] = out;
  if (npmpc_hess(g_arg, g_res, NULL, g_w, 0) != 0) abort();
}
