#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    } \
}

#define BLOCK_SIZE 16

__global__ void matrix_multiplication(float *vec1, float *vec2, float *res, size_t size) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (size_t i = 0; i < size; i += BLOCK_SIZE) {
        if (row < size && i + threadIdx.x < size)
            sA[threadIdx.y][threadIdx.x] = vec1[row * size + i + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < size && i + threadIdx.y < size)
            sB[threadIdx.y][threadIdx.x] = vec2[(i + threadIdx.y) * size + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < size && col < size)
        res[row * size + col] = sum;
}

void fill_random(float *vec, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
}

void process_matrix_multiplication(size_t size) {
    float *h_vec1, *h_vec2, *h_res;
    h_vec1 = new float[size * size];
    h_vec2 = new float[size * size];
    h_res = new float[size * size];

    fill_random(h_vec1, size * size);
    fill_random(h_vec2, size * size);

    float *d_vec1, *d_vec2, *d_res;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_vec1, size * size * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_vec2, size * size * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_res, size * size * sizeof(float)));

    CUDA_CHECK_RETURN(cudaMemcpy(d_vec1, h_vec1, size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_vec2, h_vec2, size * size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks((size + threads_per_block.x - 1) / threads_per_block.x,
                    (size + threads_per_block.y - 1) / threads_per_block.y);

    cudaEvent_t start, end;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&end));

    const int iterations = 10;
    float total_time = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK_RETURN(cudaEventRecord(start));
        matrix_multiplication<<<num_blocks, threads_per_block>>>(d_vec1, d_vec2, d_res, size);
        CUDA_CHECK_RETURN(cudaEventRecord(end));
        CUDA_CHECK_RETURN(cudaEventSynchronize(end));
        float iter_time;
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&iter_time, start, end));
        total_time += iter_time;
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(h_res, d_res, size * size * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Runtime API, Matrix Size %lux%lu: %.4f мс (усреднено за %d итераций)\n",
           size, size, total_time / iterations, iterations);

    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_res;

    CUDA_CHECK_RETURN(cudaFree(d_vec1));
    CUDA_CHECK_RETURN(cudaFree(d_vec2));
    CUDA_CHECK_RETURN(cudaFree(d_res));

    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(end));
}

int main() {
    std::vector<size_t> sizes = {256, 512, 1024, 2048, 4096, 8192};
    for (size_t size : sizes) {
        process_matrix_multiplication(size);
    }
    return 0;
}