
/* Include this file in your main.cu */

#ifndef KERNEL_H 
#define KERNEL_H
 
__global__ void msort(int *d_input, int* d_temp, int N);

#endif
