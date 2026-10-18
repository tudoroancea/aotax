#include <stdlib.h>

#include "kernel.h"
#include "layout.h"
#include "generated.h"

static generated_context* g_context;
static const double* g_inputs[NINPUTS];

void kernel_setup(const double* const* inputs) {
  if (generated_create(&g_context)) abort();
  for (int i = 0; i < NINPUTS; ++i) g_inputs[i] = inputs[i];
}

void kernel_evaluate(double* out) {
  if (generated_call(g_context, CALL_ARGS)) abort();
}
