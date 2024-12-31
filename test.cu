#include <iostream>
#include "./header/utils.cuh"
// #include "./kernel/kernel.cuh"

#define M 3  // A 的行数
#define K 3  // A 的列数 / B 的行数
#define N 3  // B 的列数

void test1(){
    // 行主序矩阵 A 和 B
    float A[M * K] = {3, 2, 1, 4, 5, 6, 7, 8, 9};  // 3x4
    float B[K * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 4x2
    float C[M * N] = {0};  // 3x2 结果矩阵

    float alpha = 1.0f;
    float beta = 0.0f;

    float *d_A, *d_B, *d_C;

    // 初始化CUDA并分配设备内存
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    testKernel(0, M, N, K, alpha, d_A, d_B, beta, d_C, handle);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // std::cout << C[i + M * j] << " ";
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);
}

// #define CEIL_DIV(a,b) ((a) + (b - 1) / (b))

void test2(){
    // 行主序矩阵 A 和 B
    float A[M * K] = {3, 2, 1, 4, 5, 6, 7, 8, 9};  // 3x4
    float B[K * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 4x2
    float C[M * N] = {0};  // 3x2 结果矩阵

    float alpha = 1.0f;
    float beta = 0.0f;

    float *d_A, *d_B, *d_C;

    // 初始化CUDA并分配设备内存
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 blockDim(32, 32);
    // dim3 gridDim(1, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    testKernel(1, M, N, K, alpha, d_A, d_B, beta, d_C, handle);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // std::cout << C[i + M * j] << " ";
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}


int main() {
    test1();
    test2();
    return 0;
}