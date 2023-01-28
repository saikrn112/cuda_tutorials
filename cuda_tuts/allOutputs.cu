#include<cuda.h>
#include <stdio.h>

__global__ void alloutputs(int *counter) {
      int oldc = atomicAdd(counter,1);
      if(*counter < 11)  printf("%d\n",oldc);
}

int main() {
  int *counter;
  cudaHostAlloc(&counter, sizeof(int),0);
  *counter = 5;
  alloutputs<<<4,3>>>(counter);
  cudaDeviceSynchronize();
  return 0;
}
