#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#define SIZE 8

//This is a sample generation of the input array. The assignment need not be evaluated on the same matrix.
void fill_array(int *mat, int M){
  for(int i=0; i<M; i++){
    // mat[i] = rand()%SIZE+1;
    mat[i] = M-i;
  }
}

// Print the array
void print_array(int *mat, int M){
  for(int i=0; i < M; i++){
    printf("%d ", mat[i]);
  }
  printf("\n");
}

int main(int argc, char** argv)
{
  // specify the dimensions of the input array
  // const int M = 32768;
  const int M = SIZE;
  unsigned numbytes = M * sizeof(int);

  int *in = (int *) malloc(numbytes);
  int *out = (int *) malloc(numbytes);

  fill_array(in, M);
  print_array(in, M); // printing the input matrix
  int *d_in, *d_out ;

  cudaError_t err;
  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out, numbytes);
  cudaMemset(d_out, 0, numbytes);
  cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

  // launching the kernel
  int numThreads = 256;
  int numBlocks = (M/numThreads)+1;
  msort<<<numBlocks, numThreads>>>(d_in, d_out, M);

  /* Print the last error encountered -- helpful for debugging */
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  //print_array(out, M);

  return 0;
}
