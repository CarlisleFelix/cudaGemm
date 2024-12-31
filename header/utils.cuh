#include <iostream>
#include <windows.h>
#include <cublas_v2.h>
#include<cuda_runtime.h>

void cudaCheck(cudaError_t error, const char *file, int line);
#define CUDA_CHECK(error) cudaCheck(error, __FILE__, __LINE__)
void cudaDeviceInfo();

void randomizeMatrix(float *mat, int M, int N);
void printMatrix(const float *mat, int M, int N);
bool compareMatrix(const float *mat1, const float *mat2, int M, int N);
void copyMatrix(const float *src, float *desc, int M, int N);

LARGE_INTEGER getFrequency();
LARGE_INTEGER getCurrentTick();
double cpuElapsedTime(LARGE_INTEGER &beg, LARGE_INTEGER &end);

void testKernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);