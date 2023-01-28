
#ifndef _KERNELS_H_
#define _KERNELS_H_
#include <stdio.h>
__global__ void transpose_parallel_per_row(float *in, float *out, int rows_in, int cols_in);


__global__ void
transpose_parallel_per_element(float *in, float *out, int rows_in, int cols_in, int K1, int K2);

#endif
