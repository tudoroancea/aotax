// What each backend's adapter provides to main.cpp: a one-time setup with the inputs, then one
// evaluation that writes the lower-triangle Hessian values, in layout.h order, to caller memory.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void kernel_setup(const double* z, const double* p, const double* lam_g);
void kernel_evaluate(double* out);

#ifdef __cplusplus
}
#endif
