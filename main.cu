#include <iostream>
#include "./header/utils.cuh"

// kernel m n k
int main(int argc, char **argv) {
    if(argc != 2) {
        std::cout << "requires 1 arguments" << std::endl;
        return 0;
    }

    int kernelNum = atoi(argv[1]);
    if(kernelNum < 0 || kernelNum > 1) {
        std::cout << "invalid kernel number" << std::endl;
        return 0;
    }

    cublasHandle_t handle;
    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasCreate failed" << std::endl;
        return 0;
    }

    float elapsedTime = 0.0;
    cudaEvent_t beg, end; // ?
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // matrix size
    const int size_len = 24;
    int SIZE[size_len];
    for (int i = 0; i < size_len; i++) {
        SIZE[i] = 256 * (i + 1);
    }
    int m, n, k, max_size;
    max_size = SIZE[size_len - 1];
    std::cout << "max_size=" << max_size << std::endl;

    float alpha = 1.0, beta = 0.0; //two arbitary input parameters，C=α*AB+β*C

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_cor = nullptr;     //host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_cor = nullptr; //device matrices

    A = new float[max_size * max_size];
    B = new float[max_size * max_size];
    C = new float[max_size * max_size];
    C_cor = new float[max_size * max_size];

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), max_size * max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), max_size * max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), max_size * max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC_cor), max_size * max_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_cor, C_cor, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeatTimes = 10;
    for (int i = 0; i < size_len; i++) {
        m = n = k = SIZE[i];
        std::cout << "m=n=k=" << k << std::endl;
        testKernel(0, m, n, k, alpha, dA, dB, beta, dC_cor, handle);
        testKernel(kernelNum, m, n, k, alpha, dA, dB, beta, dC, handle);
        cudaDeviceSynchronize();
        cudaMemcpy(C_cor, dC_cor, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if(!compareMatrix(C, C_cor, m, n)) {
            return 0;
        }
        cudaDeviceSynchronize();

        cudaEventRecord(beg);
        for (int j = 0; j < repeatTimes; j++) {
            testKernel(kernelNum, m, n, k, alpha, dA, dB, beta, dC, handle);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsedTime, beg, end);
        elapsedTime /= 1000.; //换算成秒
        std::cout << "Average elasped time: " << elapsedTime / repeatTimes << " second, performance: "
        << 2. * 1e-9 * repeatTimes * m * n * k / elapsedTime << " GFLOPS. " << "size: " << m << "." << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_cor;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_cor);
    return 0;
}