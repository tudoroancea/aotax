// The aotax library behind kernel.h, called through its generated C API.
#include <stdlib.h>

#include "kernel.h"
#include "npmpc_hess.h"

static npmpc_hess_context* g_context;
static const double *g_z, *g_p, *g_lam_g;

void kernel_setup(const double* z, const double* p, const double* lam_g) {
  if (npmpc_hess_create(&g_context)) abort();
  g_z = z;
  g_p = p;
  g_lam_g = lam_g;
}

void kernel_evaluate(double* out) {
  if (npmpc_hess_call(g_context, g_z, g_p, g_lam_g, out)) abort();
}
