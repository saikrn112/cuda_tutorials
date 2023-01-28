#include <stdio.h>
#include "timer.h"
#include "kernels.h"

//This is a sample generation of the input matrix. The assignment need not be evaluated on the same matrix.

void fill_matrix(float *mat, int M, int N)
{
  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      {
        mat[i*N +j] = i + 2.3f * j;
      }
}

// Print the matrix
void print_matrix(float *mat, int M, int N)
{
  for(int i=0; i < M; i++)
  {
    for(int j=0; j < N; j++) {
      printf("%4.3g ", mat[j + i*N]); }
    printf("\n");
  }
  printf("\n");
}

// Verify the correctness by comparing the sequential output with parallel output
bool compare_matrices(float *gpu, float *ref, int rows, int cols)
{

  for(int i=0; i < rows; i++)
  {
    for(int j=0; j < cols; j++)
    {
      if (ref[i*cols + j] != gpu[i*cols +j])
         {
           return false;
         }
    }
  }
    return true; // generated output matches expected output
}

// Generating expected output
void transpose_CPU(float in[], float out[], int M, int N)
{
  for(int i=0; i<M; i++)
  for(int j=0; j<N; j++)
           out[j*M +i] = in[i * N + j];
}


int main(int argc, char** argv)
{
  // specify the dimensions of the input matrix
  const int M = 32768; // number of rows
  const int N = 8192; // number of columns

  unsigned numbytes = M * N * sizeof(float);

  float *in = (float *) malloc(numbytes);
  float *out = (float *) malloc(numbytes);
  float *out1 = (float *) malloc(numbytes);
  float *gold = (float *) malloc(numbytes);

  fill_matrix(in, M, N);
  CPUTimer cputimer;
  cputimer.Start();
  transpose_CPU(in, gold, M, N);
  cputimer.Stop();
  printf("The sequential code ran in %f ms\n", cputimer.Elapsed()*1000);
  //print_matrix(in, M, N); // printing the input matrix
  //print_matrix(gold, N, M); // printing expected output matrix

  float *d_in, *d_out1, *d_out2 ;

  cudaError_t err;
  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out1, numbytes);
  cudaMalloc(&d_out2, numbytes);

  cudaMemset(d_out1, 0, numbytes);
  cudaMemset(d_out2, 0, numbytes);

  cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);


  GPUTimer timer;
  timer.Start();

  // launching the kernel
  int numThreads = 256;
  transpose_parallel_per_row<<<(M/numThreads)+1, numThreads>>>(d_in, d_out1, M, N);

  timer.Stop();

  /* Print the last error encountered -- helpful for debugging */
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(out, d_out1, numbytes, cudaMemcpyDeviceToHost);
  //print_matrix(out, N, M);
  printf("Result of <transpose_parallel_per_row>: %s\n",compare_matrices(out, gold, N, M) ? "Success" : "Failure"); /* Success <--> correct output */
  printf("The kernel <transpose_parallel_per_row> ran in: %f ms\n", timer.Elapsed());

  // Sample values for K1 and K2 (subject to change).
  const int K1 = 32; // should be a divisor of M
  const int K2 = 8; // should be a divisor of N

  dim3 blocks(M/K1,N/K2); // blocks per grid
  dim3 threads(K1,K2);  // threads per block

  timer.Start();

  // launching the kernel
  transpose_parallel_per_element<<<250000,100000>>>(d_in, d_out2,M,N,K1,K2);

  timer.Stop();

/* Print the last error encountered -- helpful for debugging */
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(out1, d_out2, numbytes, cudaMemcpyDeviceToHost);
  printf("Result of <transpose_parallel_per_element>: %s\n",compare_matrices(out1, gold, N, M) ? "Success" : "Failure");/* Success <--> correct output */
  //print_matrix(out1, N, M);
  printf("The kernel <transpose_parallel_per_element> ran in: %f ms\n", timer.Elapsed());

  return 0;
}
