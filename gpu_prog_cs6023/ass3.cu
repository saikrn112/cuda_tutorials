#include <cuda.h>
#include <stdio.h>
__global__ void alloutputs(int *counter) {
      //(*counter)++;
      atomicAdd(counter,1);
      __syncthreads();
      printf("%d, %d\n",threadIdx.x, *counter);
}
int main() {
    int *counter, hcounter = 0;
    cudaMalloc(&counter, sizeof(int));
    cudaMemcpy(counter, &hcounter, sizeof(int), cudaMemcpyHostToDevice);
    alloutputs<<<1, 288>>>(counter);
    cudaDeviceSynchronize();
    int *n = new int; //(int *) malloc(sizeof(int));
    cudaMemcpy(n,counter,sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n",*n);
    return 0;
}
