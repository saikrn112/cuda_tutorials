#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "kernels.h"

const int col = 60;
const int row = 50;
// const int col = 20;
// const int row = 7;

const int bin_count = 31;

void make_matrix(int *mat, int rows, int cols)
{
	srand(time(NULL));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mat[i*cols + j] = rand() % 20000;
		}
	}
}

void printt(int bin[]){
	for(int i=0;i<bin_count;i++)
		printf("%d ",bin[i]);
}

bool histogram(int *in, int *bin)
{
	int i, j,val;
	int *temp = (int *)malloc(sizeof(int)*bin_count);
	for (i = 0; i < bin_count; i++)
	{
		temp[i] = 0;
	}
	for (i = 0; i < row; i++)
	{

		for (j = 0; j < col; j++)
		{
			val = in[i*col + j] % bin_count;
			temp[val]++;
		}
	}
	 //printt(temp);
	 //printf("\n");
   //printt(bin);
	for (i = 0; i < bin_count; i++)
	{
		if (temp[i] != bin[i])
		{
			free(temp);
			return false;

		}
	}
	free(temp);
	return true;
}

void print2(int temp[]){
	for(int i=0;i<row;i++){
		printf("\n");
		for(int j=0;j<col;j++)
			printf("%d  ",temp[i*col+j]);
	}
printf("\n");
}

bool Stencil(int *in,int *temp_array)
{
	int *temp = (int *)malloc(sizeof(int)*row*col);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (i == 0 || i == row - 1 || j == 0 || j == col - 1)
				temp[i*col + j] = in[i*col + j];
			else
			{
				temp[i*col + j] = 0.2*(temp_array[i*col + j] + temp_array[i*col + j-1] + temp_array[i*col + j+1] + temp_array[(i-1)*col + j] + temp_array[(i+1)*col + j]);
			}
		}
	}
	// print2(temp);
	for (int i =0; i<row; i++){
		for (int j = 0; j <col; j++){
			if (temp[i*col + j] != in[i*col + j])
			{
				printf("\n(%d,%d) === %d, %d\n",i,j, in[i*col + j], temp[i*col + j]);
				free(temp);
				return false;

			}
		}
	}
	free(temp);
	return true;
}

bool BC(int *input)
{
	int i;

	for (i = 0; i < col; i++)
	{
		if (input[i] != 1 || input[(row - 1)*col + i] != 1)
		{
			printf("%d %d %d", input[i], input[(row - 1)*col + i],i);
			return false;
		}
	}

	for (i = 0; i < row; i++)
	{
		if (input[i*col] != 1 || input[i*col + col - 1] != 1)
		{
			printf("%d %d ", input[i*col], input[i*col + col - 1]);
			return false;
		}
	}
	return true;
}

int main(int argc, char** argv)
{

	int num_of_elements = col*row;
	unsigned numbytes = num_of_elements*sizeof(int);
	unsigned bin_size = bin_count*sizeof(int);


	int *in = (int *)malloc(numbytes);
	int *d_in;
	cudaMalloc(&d_in, numbytes);

	make_matrix(in, row, col);

	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	int *d_bin;

  int *bin_cpu  = (int *)malloc(bin_size);

	cudaMalloc(&d_bin, bin_size);

	cudaMemset(d_bin, 0, bin_size);

	dim3 grid(4, 4);
	dim3 blocks( 16, 16);
	// dim3 grid(2, 2);
	// dim3 blocks( 10, 10);

	// dim3 threads(10,10);
	int * temp_array = (int *)malloc(numbytes);
	for (int k = 0; k < row; k++)
	{
		for (int q = 0; q < col; q++)
		{
			temp_array[k*col + q] = in[k*col + q];
		}
	}


	cudaMemcpy(in, d_in, numbytes, cudaMemcpyDeviceToHost);

// 	histogram <<<row, col >>> (d_in, d_bin, row, col, bin_count);
// 	cudaMemcpy(bin_cpu, d_bin, bin_size, cudaMemcpyDeviceToHost);
// 	printf("\nResult of histogram: %s\n", histogram(temp_array, bin_cpu) ? "Success" : "Failure");
// cudaDeviceSynchronize();
// 	updateBC << <grid, blocks >> > (d_in, row, col);
// 	cudaMemcpy(in, d_in, numbytes, cudaMemcpyDeviceToHost);
// 	printf("\nResult of UpdateBC: %s\n", BC(in) ? "Success" : "Failure");
// 	cudaDeviceSynchronize();
	// cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);
	// print2(temp_array);
	stencil << <grid,blocks>> > (d_in, row, col);
	cudaMemcpy(in, d_in, numbytes, cudaMemcpyDeviceToHost);
	printf("\nResult of Stencil: %s\n", Stencil(in, temp_array) ? "Success" : "Failure");
	// print2(in);

	cudaFree(d_bin);
	cudaFree(d_in);

	free(temp_array);
	free(in);


	return 0;
}
