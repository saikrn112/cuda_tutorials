#include <cuda.h>
#include <stdio.h>
#include "timer.h"


__global__ void coalescing(int * arr, int param) {
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int timeSteps = 32 - param + 1; // Time steps required to calculate how many  access at a time

  for(int i = 0; i < timeSteps; i++){
    if(id >= i and id < param + i){ // Checks if the thread id is with in the range for accessing memory from array at that particular time Step
       arr[id]++; // Process something with the fetched memory
    }
  }
}

int main() {
    int N = 32; // Array of size 32
    int *counter;
    int *hcounter = (int*)malloc(N*sizeof(int));
    int i = 0;
    for(i = 0; i< N; i++){
      hcounter[i] = i;
    }

    cudaMalloc(&counter, N*sizeof(int));
    cudaMemcpy(counter, &hcounter, sizeof(int), cudaMemcpyHostToDevice);
    GPUTimer timer;
    dim3 threads(55,55);
    float t = 0;
    for(i = 0; i< 10000; i++){
      timer.Start();
      coalescing<<<1,threads>>>(counter, 32);
      cudaDeviceSynchronize();
      timer.Stop();
      t += 1000*timer.Elapsed();
    }

    printf("GPU elapsed %f ms \n", t/10000);
    return 0;
}
