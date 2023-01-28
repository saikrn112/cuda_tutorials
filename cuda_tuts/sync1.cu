#include<cuda.h>
#include <stdio.h>
__device__ volatile int n;

__global__ void oneToTen() {
      __shared__ int n;
      n = 0;
      //__syncthreads();
      while(n <10){
        if(n%3 == threadIdx.x){
          printf("%d: %d\n",threadIdx.x,n );
          ++n;
        }
        __syncthreads();
      }
      // for(int ii=0; ii <10; ii++){
      //   if(ii%3==threadIdx.x){
      //     printf("%d: %d\n",threadIdx.x,ii );
      //   }
      // }
}
int main() {
    oneToTen<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
