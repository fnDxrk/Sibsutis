#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <ctime>

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}

__global__ void dot_product_kernel(int *a, int *b, int *partial_result, int n) {
    extern __shared__ int cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    int temp = 0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s) {
            cache[cacheIndex] += cache[cacheIndex + s];
        }
        __syncthreads();
    }

    if (cacheIndex == 0) {
        partial_result[blockIdx.x] = cache[0];
    }
}

void benchmark_dot_product(size_t N, size_t chunk_size) {
    int *h_a = new int[N];
    int *h_b = new int[N];
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    size_t num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<cudaStream_t> streams(num_chunks);

    int block_size = 256;
    int total_result = 0;

    std::vector<int*> d_a_chunks(num_chunks), d_b_chunks(num_chunks), d_partial_chunks(num_chunks);
    std::vector<int*> h_partial_chunks(num_chunks);
    std::vector<int> partial_sizes(num_chunks);

    auto start = std::chrono::high_resolution_clock::now();

    // Создаем стримы
    for (size_t i = 0; i < num_chunks; ++i) {
        CUDA_CHECK_RETURN(cudaStreamCreate(&streams[i]));
    }

    for (size_t i = 0; i < num_chunks; ++i) {
        size_t offset = i * chunk_size;
        size_t current_chunk = std::min(chunk_size, N - offset);
        partial_sizes[i] = (current_chunk + block_size - 1) / block_size;

        CUDA_CHECK_RETURN(cudaMalloc(&d_a_chunks[i], current_chunk * sizeof(int)));
        CUDA_CHECK_RETURN(cudaMalloc(&d_b_chunks[i], current_chunk * sizeof(int)));
        CUDA_CHECK_RETURN(cudaMalloc(&d_partial_chunks[i], partial_sizes[i] * sizeof(int)));
        h_partial_chunks[i] = new int[partial_sizes[i]];

        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_a_chunks[i], h_a + offset,
                            current_chunk * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_b_chunks[i], h_b + offset,
                            current_chunk * sizeof(int), cudaMemcpyHostToDevice, streams[i]));

        dot_product_kernel<<<partial_sizes[i], block_size, block_size * sizeof(int), streams[i]>>>(
            d_a_chunks[i], d_b_chunks[i], d_partial_chunks[i], current_chunk
        );

        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_partial_chunks[i], d_partial_chunks[i],
                            partial_sizes[i] * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

    for (size_t i = 0; i < num_chunks; ++i) {
        CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i]));
        for (int j = 0; j < partial_sizes[i]; ++j) {
            total_result += h_partial_chunks[i][j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Chunk size: %zu KB (%zu elements)\n", chunk_size / 1024, chunk_size);
    printf("Time: %.4f ms\n\n", elapsed.count());

    for (size_t i = 0; i < num_chunks; ++i) {
        CUDA_CHECK_RETURN(cudaFree(d_a_chunks[i]));
        CUDA_CHECK_RETURN(cudaFree(d_b_chunks[i]));
        CUDA_CHECK_RETURN(cudaFree(d_partial_chunks[i]));
        delete[] h_partial_chunks[i];
        CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i]));
    }

    delete[] h_a;
    delete[] h_b;
}

int main() {
    srand(time(nullptr));
    const size_t N = 1 << 24;
    printf("Vector size: %zu elements\n", N);

    std::vector<size_t> chunk_sizes = {
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24 
    };

    for (size_t chunk : chunk_sizes) {
        benchmark_dot_product(N, chunk);
    }

    return 0;
}
