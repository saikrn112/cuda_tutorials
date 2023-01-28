
/* Include this file in your main.cu */

#ifndef KERNEL_H 
#define KERNEL_H

__global__ void histogram(int *d_input, int* d_bin, int M, int N, int BIN_COUNT);
__global__ void stencil(int* d_input, int M, int N);
__global__ void updateBC(int* d_input, int M, int N);

#endif
