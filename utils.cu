#include <iostream>
#include <iomanip>
#include <random>
#include <windows.h>
#include <cmath>
#include "header/utils.cuh"
#include "kernel/kernel.cuh"

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void cudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    std::cout << "Device ID: " << deviceId << std::endl;
    std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
    std::cout << "Compute Capability Major: " << props.major << std::endl;
    std::cout << "Compute Capability Minor: " << props.minor << std::endl;
    std::cout << "memoryBusWidth: " << props.memoryBusWidth << std::endl;
    std::cout << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "totalGlobalMem: " << props.totalGlobalMem / 1024 / 1024 << std::endl;
    std::cout << "sharedMemPerBlock: " << props.sharedMemPerBlock / 1024 << std::endl;
    std::cout << "sharedMemPerMultiprocessor: " << props.sharedMemPerMultiprocessor / 1024 << std::endl;
    std::cout << "totalConstMem: " << props.totalConstMem / 1024 << std::endl;
    std::cout << "multiProcessorCount: " << props.multiProcessorCount << std::endl;
    std::cout << "Warp Size: " << props.warpSize << std::endl;
};

void randomizeMatrix(float *mat, int M, int N) {
    std::random_device rd; // 获取硬件随机数种子
    std::mt19937 gen(rd()); // 使用梅森旋转算法生成随机数
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f); // 生成0到1之间的随机浮点数
    for (int i = 0; i < M * N; i++) {
        float tmp = dis(gen);
        mat[i] = tmp;
    }
}

void printMatrix(const float *A, int M, int N) {
    int i;
    std::cout << "[";
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            std::cout << std::fixed << std::setprecision(2) << A[i] << " ";
        else
            std::cout << std::fixed << std::setprecision(2) << A[i] << ", ";
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                std::cout << ";" << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}

bool compareMatrix(const float *A, const float *B, int M, int N) {
    // for (int i = 0; i < 5; ++i) {
    //     std::cout << "line " << i << std::endl;
    //     std::cout << "A: " << std::endl;
    //     for (int j = 0; j < 5; ++j) {
    //         // std::cout << C[i + M * j] << " ";
    //         // std::cout << A[i * N + j] << " ";
    //         std::cout << std::fixed << std::setprecision(2) << A[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "B: " << std::endl;
    //     for (int j = 0; j < 5; ++j) {
    //         // std::cout << C[i + M * j] << " ";
    //         // std::cout << B[i * N + j] << " ";
    //         std::cout << std::fixed << std::setprecision(2) << B[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }


    for(int i = 0; i < M * N; i++) {
         if(fabs(A[i] - B[i]) > 1e-2){
             std::cout << "loss too big at " << i << std::endl;
             return false;
         }
    }
    return true;
}

void copyMatrix(const float *src, float *desc, int M, int N) {
    for(int i = 0; i < M * N; i++) {
        desc[i] = src[i];
    }
}

LARGE_INTEGER getFrequency(){
     static bool initialized = false;
     static LARGE_INTEGER frequency;
     if(!initialized){
         QueryPerformanceFrequency(&frequency);
     }
     return frequency;
}

LARGE_INTEGER getCurrentTick(){
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return counter;
}

double cpuElapsedTime(LARGE_INTEGER &beg, LARGE_INTEGER &end){
    return static_cast<double>(end.QuadPart - beg.QuadPart) / getFrequency().QuadPart;
}

#define CEIL_DIV(a,b) ((a) + (b - 1) / (b))

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void test_mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    gemmv1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void testKernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C,
                 cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            test_cublas(handle, M, N, K, alpha, A, B, beta, C);
        break;
        case 1:
            test_mysgemm_v1(M, N, K, alpha, A, B, beta, C);
        break;
        case 2:
            // not implemented
        break;
        case 3:
            // not implemented
        break;
        case 4:
            // not implemented
        break;
        case 5:
            // not implemented
        break;
        case 6:
            // not implemented
        break;
        case 7:
            // not implemented
        break;
        default:
            break;
    }
}