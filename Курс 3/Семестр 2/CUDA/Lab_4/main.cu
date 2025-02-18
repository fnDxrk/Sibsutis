#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    // Длина векторов: 1 << 20 = 1048576 элементов
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(reinterpret_cast<void**>(&d_A), size);
    cudaMalloc(reinterpret_cast<void**>(&d_B), size);
    cudaMalloc(reinterpret_cast<void**>(&d_C), size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSizes[] = { 1, 16, 32, 64, 128, 256, 512, 1024 };
    int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);

    std::cout << "Выполнение ядра в зависимости от числа нитей в блоке:" << std::endl;

    for (int i = 0; i < numConfigs; i++) {
        int threadsPerBlock = blockSizes[i];
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        cudaEventRecord(start, 0);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Threads per Block: " << threadsPerBlock
                  << ", Blocks per Grid: " << blocksPerGrid
                  << ", Time: " << milliseconds << " ms" << std::endl;
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - 3.0f) > 1e-5) {
            std::cerr << "Ошибка в элементе " << i << ": " << h_C[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Вычисления прошли успешно!" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
