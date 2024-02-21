#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>

__global__ void alloutputs(int *counter) {
      int oldc = atomicAdd(counter,1);
      if(*counter < 11)  printf("%d\n",oldc);
}

__global__ void alloutputs2(int* d_vec, int N)
{
    int row = threadIdx.x;
    if (row >= N)
    {
        return; // Exit if the thread index is out of range
    }
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            // sdata[threadIdx.x] += sdata[threadIdx.x + s];
            d_vec[threadIdx.x] = atomicMax(&d_vec[threadIdx.x], d_vec[threadIdx.x + s]);
        }
        __syncthreads();
    }
    // for (int stride = 1; stride < N; stride *= 2)
    // {
    //     int index = 2 * stride * row;
    //     if (index < blockDim.x) {
    //     }

    //     __syncthreads(); // Synchronize threads after each iteration
    // }
}



int main() {
    std::srand(42);
    int N = 10;
    std::vector<int> h_vec(N,0);
    for (int i = 0; i < N; i++)
    {
        h_vec[i] = std::rand() %1000;
    }
    // Find the maximum element using std::max_element
    auto max_element_iter = std::max_element(h_vec.begin(), h_vec.end());

    // Check if the iterator is valid before dereferencing it
    if (max_element_iter != h_vec.end()) {
        std::cout << "std max value: " << *max_element_iter << std::endl;
    } else {
        std::cout << "Vector is empty." << std::endl;
    }


    int *d_vec;
    cudaMalloc(&d_vec, sizeof(int)*N);
    cudaMemcpy(d_vec, h_vec.data(), sizeof(int)*N, cudaMemcpyHostToDevice);

    alloutputs2<<<N/2,1>>>(d_vec, N);

    int* val;
    cudaMallocManaged(&val, sizeof(int)); // Allocate memory for val
    cudaMemcpy(val, d_vec, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "max value: " << *val << std::endl;
    // std::cout << "max value:" << *val << std::endl;

    for (int i = 0; i < N; i++)
    {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    // int *counter;
    // cudaHostAlloc(&counter, sizeof(int),0);
    // *counter = 5;
    // alloutputs<<<4,3>>>(counter);
    // cudaDeviceSynchronize();
  return 0;
}
