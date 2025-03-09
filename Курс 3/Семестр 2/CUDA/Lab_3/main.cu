#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx == 3) {
            int* ptr = NULL;
            // Неправильное обращение к памяти: разыменование NULL.
            ptr[0] = 0;
        }
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = 10;
    size_t size = N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err;

    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Ошибка выделения памяти на устройстве для A: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Ошибка выделения памяти на устройстве для B: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Ошибка выделения памяти на устройстве для C: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Ошибка выполнения CUDA ядра: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Результирующий вектор:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
