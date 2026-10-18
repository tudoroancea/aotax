#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void kernel_setup(const double* const* inputs);
void kernel_evaluate(double* out);

#ifdef __cplusplus
}
#endif
