#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <chrono>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)


template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

// Function to convert std::vector<std::vector<int>> to raw C-style arrays
void vectorToRawArrays(const std::vector<std::vector<int>>& input,
                       int*& dataArray, // Pointer to flattened array
                       int& rows,       // Number of rows
                       int& cols)       // Number of columns
{
    // Get the number of rows and columns
    rows = input.size();
    cols = (rows > 0) ? input[0].size() : 0;

    // Allocate memory for the flattened array
    dataArray = new int[rows * cols];

    // Copy data from std::vector<std::vector<int>> to the flattened array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dataArray[i * cols + j] = input[i][j];
        }
    }
}

// Function to free memory allocated for the raw C-style arrays
void freeRawArrays(int* dataArray)
{
    delete[] dataArray;
}

std::vector<std::vector<int>> rawArraysToVector(int*& h_matrixC, int rowsC, int colsC)
{
    std::vector<std::vector<int>> matrixC(rowsC, std::vector<int>(colsC));
    int index = 0; // Index to iterate over the 1D array

    for (int i = 0; i < rowsC; i++)
    {
        for (int j = 0; j < colsC; j++)
        {
            // Access the element at the current index in the 1D array
            matrixC[i][j] = h_matrixC[index++];
        }
    }
    return matrixC;
}


__global__ void cuda_matmul(int* d_A, 
            int* d_B,
            int* d_C,
            size_t rowsA, size_t colsA, size_t rowsB, 
            size_t colsB, size_t rowsC, size_t colsC)
{

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row >= rowsC || col >= colsC)
    {
        return;
    }

    long int sum = 0;
    for (size_t k = 0; k < colsA; k++ )
    {
        sum += d_A[row*colsA + k]*d_B[k*colsB + col];
    }
    d_C[row*colsC + col] = sum;

}

#define TILE_WIDTH  2

__global__ void cuda_matmul_tiling(int* d_A, 
            int* d_B,
            int* d_C,
            size_t rowsA, size_t colsA, size_t rowsB, 
            size_t colsB, size_t rowsC, size_t colsC)
{
    __shared__ int d_As[TILE_WIDTH][TILE_WIDTH];
    __shared__ int d_Bs[TILE_WIDTH][TILE_WIDTH];

    // int row = blockIdx.y*blockDim.y + threadIdx.y;
    // int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x*TILE_WIDTH + threadIdx.x;

    if (row >= rowsC || col >= colsC)
    {
        return;
    }
    int sum = 0;
    for (size_t ph = 0; ph < colsC/TILE_WIDTH; ph++)
    {
        d_As[threadIdx.y][threadIdx.x] = d_A[row*colsA + ph*TILE_WIDTH + threadIdx.x];
        d_Bs[threadIdx.y][threadIdx.x] = d_B[(ph*TILE_WIDTH + threadIdx.y)*colsB + col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            sum += d_As[threadIdx.y][k]*d_Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    d_C[row*colsC + col] = sum;

    // tiling
    // have a tiling width
    // initialize the shared memory based on tiling width
    // load the corresponding elements into shared memory
    // 


    // for (size_t k = 0; k < colsA; k++ )
    // {
    //     d_C[row*colsA + col] += d_A[row*colsA + k]*d_B[k*colsB + col];
    // }
}

// __global__ void cuda_matmul_tiling(int* d_A, 
//             int* d_B,
//             int* d_C,
//             size_t rowsA, size_t colsA, size_t rowsB, 
//             size_t colsB, size_t rowsC, size_t colsC)
// {
//     __shared__ d_As[TILE_WIDTH][TILE_WIDTH];
//     __shared__ d_Bs[TILE_WIDTH][TILE_WIDTH];

//     auto row = blockIdx.x * TILE_WIDTH
// }


// __global__ void cuda_matmul_tiling(int* d_A, int* d_B, int* d_C, size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, size_t rowsC, size_t colsC) {
//     __shared__ int d_As[TILE_WIDTH][TILE_WIDTH];
//     __shared__ int d_Bs[TILE_WIDTH][TILE_WIDTH];

//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row >= rowsC || col >= colsC) return;
//     int sum = 0;
//     for (int ph = 0; ph < ceil(colsA / (float)TILE_WIDTH); ++ph) {
//         if ((row < rowsA) && (ph*TILE_WIDTH + threadIdx.x < colsA))
//             d_As[threadIdx.y][threadIdx.x] = d_A[row * colsA + ph * TILE_WIDTH + threadIdx.x];
//         else
//             d_As[threadIdx.y][threadIdx.x] = 0;

//         if ((col < colsB) && (ph*TILE_WIDTH + threadIdx.y < rowsB))
//             d_Bs[threadIdx.y][threadIdx.x] = d_B[(ph * TILE_WIDTH + threadIdx.y) * colsB + col];
//         else
//             d_Bs[threadIdx.y][threadIdx.x] = 0;

//         __syncthreads();

//         for (int k = 0; k < TILE_WIDTH; ++k) {
//             sum += d_As[threadIdx.y][k] * d_Bs[k][threadIdx.x];
//         }
//         __syncthreads();
//     }
//     if (row < rowsC && col < colsC)
//         d_C[row * colsC + col] = sum;
// }

void processesInCudaCore(
            int* h_matrixA, 
            int* h_matrixB,
            int*& h_matrixC,
            size_t rowsA, size_t colsA, size_t rowsB, 
            size_t colsB, size_t rowsC, size_t colsC)
{
    assert(colsA == rowsB);
    assert(rowsA == rowsC);
    assert(colsB == colsC);

    // push from host to device 
    int *d_A, *d_B, *d_C; 
    checkCudaErrors(cudaMalloc(&d_A, sizeof(int)*rowsA*colsA));
    checkCudaErrors(cudaMalloc(&d_B, sizeof(int)*rowsB*colsB));
    checkCudaErrors(cudaMalloc(&d_C, sizeof(int)*rowsC*colsC));
    checkCudaErrors(cudaMemset(d_A, 0, sizeof(int)*rowsA*colsA));
    checkCudaErrors(cudaMemset(d_B, 0, sizeof(int)*rowsB*colsB));
    checkCudaErrors(cudaMemset(d_C, 0, sizeof(int)*rowsC*colsC));
    checkCudaErrors(cudaMemcpy(d_A, h_matrixA, sizeof(int)*rowsA*colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_matrixB, sizeof(int)*rowsB*colsB, cudaMemcpyHostToDevice));
    
    // h_matrixC = const_cast<int*>(h_matrixA);
    

    // compute kernel 
    dim3 blockSize (TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridSize (rowsC/blockSize.x + 1,colsC/blockSize.y + 1,1); 


    // dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridSize((colsC + blockSize.x - 1) / blockSize.x, (rowsC + blockSize.y - 1) / blockSize.y, 1);

    auto startCuda = std::chrono::steady_clock::now();
    cuda_matmul_tiling<<<gridSize, blockSize>>>(d_A,d_B,d_C, rowsA, colsA, 
            rowsB, colsB, rowsC, colsC);
    cudaDeviceSynchronize();
    auto endCuda = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedCuda = endCuda - startCuda;
    std::cout << "CUDA Inner Multiplication Time: " << elapsedCuda.count() << " seconds" << std::endl;


    // copy from device to host
    checkCudaErrors(cudaMemcpy(h_matrixC, d_C, sizeof(int)*rowsC*colsC, cudaMemcpyDeviceToHost));
    
}



std::vector<std::vector<int>> processesInCuda(
            const std::vector<std::vector<int>>& matrixA, 
            const std::vector<std::vector<int>>& matrixB)
{
    int* h_matrixA = nullptr; int rowsA; int colsA; 
    int* h_matrixB = nullptr; int rowsB; int colsB;
    vectorToRawArrays(matrixA, h_matrixA, rowsA, colsA);
    vectorToRawArrays(matrixB, h_matrixB, rowsB, colsB);

    int* h_matrixC = nullptr; int rowsC = rowsA; int colsC = colsB;
    h_matrixC = new int[rowsC*colsC];
    processesInCudaCore(h_matrixA, h_matrixB, h_matrixC, 
                rowsA, colsA, rowsB, colsB, rowsC, colsC);
    
    auto matrixC = rawArraysToVector(h_matrixC, rowsC, colsC);
    freeRawArrays(h_matrixA);
    freeRawArrays(h_matrixB);
    freeRawArrays(h_matrixC);
    return matrixC;

}