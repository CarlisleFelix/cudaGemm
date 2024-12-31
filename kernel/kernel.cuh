#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__
void gemmv1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    // int
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;


    if(tidx < M && tidy < N){
        float tmp = 0;
        for(int i = 0; i < K; i++){
            tmp += A[tidx * K + i] * B[i * N + tidy];
        }
        C[tidx * N + tidy] = alpha * tmp + beta * C[tidx * N + tidy];
    }
}