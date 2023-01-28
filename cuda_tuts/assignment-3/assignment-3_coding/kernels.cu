#include "kernels.h"
#include <stdio.h>

 __device__  int counter = 0;

__global__ void stencil(int* d_input, int M, int N){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int idy = threadIdx.y + blockIdx.y*blockDim.y;
  int Aij = 0, A_ij = 0, Ai__j = 0, Ai_j = 0, Aij_ = 0;

  if(idx > 0  and idx < M-1 and idy > 0 and idy < N-1){
    Aij = d_input[idx*N + idy];
    Ai_j = d_input[(idx+1)*N + (idy)];
    A_ij = d_input[(idx-1)*N + (idy)];
    Aij_ = d_input[(idx)*N + (idy + 1)];
    Ai__j = d_input[(idx)*N + (idy - 1)];

  }
  __syncthreads();
  // block syncing;
  if(threadIdx.x == 0 and threadIdx.y == 0){
    atomicAdd(&counter,1);
  //  printf("%d,(%d,%d)\n", atomicAdd(&counter,1),blockIdx.x,blockIdx.y);
  }


  while(counter != gridDim.x*gridDim.y){
    __threadfence();
  //  printf("waiting %d,%d", blockIdx.x, blockIdx.y);
  }

  if(idx > 0  and idx < M-1 and idy > 0 and idy < N-1){
    d_input[idx*N + idy] = 0.2*(Aij + Ai_j + A_ij + Aij_ + Ai__j);
  }
  __syncthreads();

}






__global__ void histogram(int *d_input, int* d_bin, int M, int N, int BIN_COUNT){
  // TODO: Your implementation goes here
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if(id < M*N){
    atomicAdd(&d_bin[(d_input[id]%BIN_COUNT)],1);
  }

}

__global__ void updateBC(int* d_input, int M, int N){

  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int idy = threadIdx.y + blockIdx.y*blockDim.y;

  if(idx < M and idy < N){
    if((idx == 0) or (idx == M-1)){
      d_input[idx*N + idy] = 1;
    } else if((idy == 0 and idx >0 and idx < M-1) or (idy == N-1 and idx < M-1 and idx >0)){
      d_input[idx*N + idy] = 1;
    }
  }
}
