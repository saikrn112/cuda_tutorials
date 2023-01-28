#include <stdio.h>
#include <cuda.h>
#include <limits.h>
using namespace std;


__global__ void saddleFinder(const int *arr, int N){
  int id = threadIdx.x;
  int max = INT_MIN, min = INT_MAX; // Using limits.h
  int i1 = 0, i2 = 0 ;

  __syncthreads(); // It is not required

  // Finding minimum in a row
  for(int i = 0; i < N; i++){
    if(min > arr[id*N+i] ){
      min = arr[id*N + i]; // Storing the minimum for checking in next iteration
      i1 = i;              // Storing the coloumn where minimum is found;
      //printf("threadIdx: %d, min: %d, i1: %d\n", id, min, i1);
    }
  }

  __syncthreads();// Not required


  // Finding Maximum in the coloumn corresponding to found minimum (in i1)
  for(int i = 0; i < N; i++){
    if(max < arr[i*N + i1]){
      max =  arr[i*N + i1]; // Storing the Maximum
      i2 = i;               // Storing the row where Maximum is found
      //printf("threadIdx: %d, max: %d, i2: %d\n", id, max, i2);
    }
  }

  // Check if the row matches with corresponding thread id (which determines the saddle condition)
  if (i2 == id){
    printf("Saddle Found: %d, at (%d,%d)\n" , arr[id*N + i1], id, i1 );
  }
}


int main(){
  // specify the dimensions of the input matrix
  const int M = 3; // square matrix dimensionss
  unsigned numbytes = M * M * sizeof(int);

  int arr[M][M] = {{1,2,4},{-1,-2,2},{4,5,6}}; // 3x3 matrix; Saddle at (2,0) == 4

  int *out;
  cudaMalloc(&out, numbytes);
  cudaMemcpy(out, arr, numbytes, cudaMemcpyHostToDevice);

  saddleFinder<<<1,M>>>(out, M);
  cudaDeviceSynchronize();
  return 0;
}
