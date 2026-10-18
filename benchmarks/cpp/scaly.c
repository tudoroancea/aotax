#include <stdlib.h>

#include "kernel.h"
#include "layout.h"
#include "generated.h"

static const double* g_arg[GENERATED_SZ_ARG];
static double* g_res[GENERATED_SZ_RES];
static int g_iw[GENERATED_SZ_IW > 0 ? GENERATED_SZ_IW : 1];
static double g_w[GENERATED_SZ_W > 0 ? GENERATED_SZ_W : 1];

void kernel_setup(const double* const* inputs) {
  for (int i = 0; i < NINPUTS; ++i) g_arg[i] = inputs[i];
}

void kernel_evaluate(double* out) {
  g_res[0] = out;
  if (GENERATED(g_arg, g_res, g_iw, g_w, 0)) abort();
}
