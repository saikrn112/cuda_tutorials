#include <cuda_runtime.h>
#include "kernels.h"

__global__ void transpose_parallel_per_row(float *in, float *out, int rows_in, int cols_in)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if (id < rows_in){
    int row = id;
    int j = 0;
    while(j<cols_in){
      out[j*rows_in + row] = in[row*cols_in + j];
      j++;
    }
  }
}

__global__ void
transpose_parallel_per_element(float *in, float *out, int rows_in, int cols_in, int K1, int K2)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int idy = threadIdx.y + blockIdx.y*blockDim.y;

  if(idx < rows_in){
    if(idy < cols_in){
      out[idy*rows_in + idx] = in[idx*cols_in + idy];
    }
  }
}
