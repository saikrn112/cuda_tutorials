#include <cuda.h>
#include <stdio.h>

__global__ void square (float *d_out, float *d_in)
{
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f;
}

int main(int argc, char ** argv)
{
    const int ARR_SIZE = 64;
    const int ARR_BYTES = ARR_SIZE * sizeof (float);


    // generate the input array on the host
    float h_in[ARR_SIZE];
    for (int i = 0; i < ARR_SIZE; i++)
    {
        h_in[i] = float(i);
    }

    float h_out[ARR_SIZE];


    //declare GPU memory pointers
    float * d_in;
    float * d_out;


    // allocate GPU memory
    cudaMalloc( (void**) &d_in, ARR_BYTES);
    cudaMalloc( (void**) &d_out, ARR_BYTES);

    // transfer the array to the GPU
    cudaMemcpy( d_in, h_in, ARR_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    square <<< 1, ARR_SIZE >>> (d_out, d_in);

    // transfer the array to the GPU
    cudaMemcpy( h_out, d_out, ARR_BYTES, cudaMemcpyDeviceToHost);


    // print out the resulting array
    for (int i = 0; i < ARR_SIZE; i++)
    {
        printf( "%f", h_out[i]);
        printf( ((i%4) != 3) ? "\t" : "\n");
    }

    // free GPU mem allocation
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
