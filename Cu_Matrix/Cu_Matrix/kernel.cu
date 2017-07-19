
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <Windows.h>

#define LINE 2048  //2048*2048耗时：0.013704s     4098*4096耗时：0.05545s
#define BLOCK_SIZE 16

__global__ void matrix_kernel_1(float* dev_c, const float* dev_a, const float *dev_b)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	
	int aStart = LINE*(by*BLOCK_SIZE*2);   
	int aEnd = aStart + LINE - 1;
	int aDiff = BLOCK_SIZE*2;

	int bStart = BLOCK_SIZE*2*bx;
	int bDiff = BLOCK_SIZE*2*LINE;

	float cSub0 = 0;
	float cSub1 = 0;
	float cSub2 = 0;
	float cSub3 = 0;
	for (int a = aStart, b = bStart; a <= aEnd; a += aDiff, b += bDiff)
	{
		__shared__ float As[BLOCK_SIZE*2][BLOCK_SIZE*2];
		__shared__ float Bs[BLOCK_SIZE*2][BLOCK_SIZE*2];

		As[ty * 2][tx * 2] = dev_a[a + LINE*ty * 2 + tx * 2];
		As[ty * 2][tx * 2 + 1] = dev_a[a + LINE*ty * 2 + tx * 2 + 1];
		As[ty * 2 + 1][tx * 2] = dev_a[a + LINE*(ty * 2 + 1) + tx * 2];
		As[ty * 2 + 1][tx * 2 + 1] = dev_a[a + LINE*(ty * 2 + 1) + tx * 2 + 1];
		
		Bs[tx*2][ty*2] = dev_b[b + LINE*ty*2 + tx*2];
		Bs[tx * 2][ty * 2 + 1] = dev_b[b + LINE*(ty * 2 + 1) + tx * 2];
		Bs[tx * 2 + 1][ty * 2] = dev_b[b + LINE*ty * 2 + tx * 2 + 1];
		Bs[tx * 2 + 1][ty * 2 + 1] = dev_b[b + LINE*(ty * 2 + 1) + tx * 2 + 1];
		//将B矩阵进行转置
		/*float TempSwap;
		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			for (int j = 0; j < BLOCK_SIZE; j++)
			{
				TempSwap = Bs[i][j];
				Bs[i][j] = Bs[j][i];
				Bs[j][i] = TempSwap;
			}
		}*/

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			cSub0 += As[ty*2][k] * Bs[tx*2][k];
			cSub1 += As[ty * 2][k] * Bs[tx * 2 + 1][k];
			cSub2 += As[ty * 2 + 1][k] * Bs[tx * 2][k];
			cSub3 += As[ty * 2 + 1][k] * Bs[tx * 2 + 1][k];
		}
		__syncthreads();
	}

	int Index0 = (by*BLOCK_SIZE*2 + ty*2)*LINE + (bx*BLOCK_SIZE*2 + tx*2);
	int Index1 = (by*BLOCK_SIZE * 2 + ty * 2)*LINE + (bx*BLOCK_SIZE * 2 + tx * 2 + 1);
	int Index2 = (by*BLOCK_SIZE * 2 + ty * 2 + 1)*LINE + (bx*BLOCK_SIZE * 2 + tx * 2);
	int Index3 = (by*BLOCK_SIZE * 2 + ty * 2 + 1)*LINE + (bx*BLOCK_SIZE * 2 + tx * 2 + 1);
	dev_c[Index0] = cSub0;
	dev_c[Index1] = cSub1;
	dev_c[Index2] = cSub2;
	dev_c[Index3] = cSub3;
}

int main()
{
	LARGE_INTEGER tc, start, stop;
	float *Matrix_A;
	float *Matrix_B;
	float *Matrix_C;
	Matrix_A = (float *)malloc(sizeof(float) * LINE * LINE);
	Matrix_B = (float *)malloc(sizeof(float) * LINE * LINE);
	Matrix_C = (float *)malloc(sizeof(float) * LINE * LINE);
	for (int i = 0; i < LINE * LINE; i++)
	{
		//Matrix_A[i] = std::rand() % 1000;
		//Matrix_B[i] = std::rand() % 1000;
		Matrix_A[i] = i % 4;
		Matrix_B[i] = i % 4;
	}

	/*for (int i = 0; i < LINE; i++)
	{
		for (int j = 0; j < LINE; j++)
		{
			std::cout << Matrix_A[i] << " ";
		}
		std::cout << std::endl;
	}*/
	float *dev_m_A;
	float *dev_m_B;
	float *dev_m_C;

	//std::chrono::system_clock::time_point GPU_start = std::chrono::system_clock::now();
	cudaMalloc((void **)(&dev_m_A), LINE *LINE*sizeof(float));
	
	cudaMalloc((void **)(&dev_m_B), LINE * LINE *sizeof(float));
	
	cudaMalloc((void **)(&dev_m_C), LINE*LINE*sizeof(float));
	
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&start);
	cudaMemcpy(dev_m_A, Matrix_A, sizeof(float) * LINE * LINE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_m_B, Matrix_B, sizeof(float) * LINE * LINE, cudaMemcpyHostToDevice);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(LINE / (BLOCK_SIZE * 2), LINE / (BLOCK_SIZE * 2));
	//dim3 blocks(1,1);
	//Calculate_Matrix <<<blocks, threads >>>(dev_m_A, dev_m_B, dev_m_C);
	matrix_kernel_1 << <blocks, threads >> >(dev_m_C, dev_m_A, dev_m_B);
	cudaMemcpy(Matrix_C, dev_m_C, sizeof(float) * LINE *LINE, cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&stop);
	
	printf("Use Time:%f\n", (stop.QuadPart - start.QuadPart)*1.0 / tc.QuadPart);
	//std::chrono::system_clock::time_point GPU_end = std::chrono::system_clock::now();
	//std::cout << double(std::chrono::duration_cast<std::chrono::microseconds>(GPU_end - GPU_start).count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s" << std::endl;
	/*for (int i = 0; i < LINE; i++)
	{
		for (int j = 0; j < LINE; j++)
		{
			std::cout << Matrix_C[i*LINE + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/
	cudaFree(dev_m_A);
	cudaFree(dev_m_B);
	cudaFree(dev_m_C);
	free(Matrix_A);
	free(Matrix_B);
	free(Matrix_C);

	system("pause");
	return 0;
}

