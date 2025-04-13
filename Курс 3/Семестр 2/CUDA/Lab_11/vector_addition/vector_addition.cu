#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}


__global__ void vector_addition_kernel(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void benchmark_with_chunk_size(size_t N, size_t chunk_size) {
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];

    for (size_t i = 0; i < N; ++i) {
        h_a[i] = rand() % 1000;
        h_b[i] = rand() % 1000;
    }

    int *d_a, *d_b, *d_c;
    CUDA_CHECK_RETURN(cudaMalloc(&d_a, N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_c, N * sizeof(int)));

    const int num_streams = (N + chunk_size - 1) / chunk_size;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK_RETURN(cudaStreamCreate(&streams[i]));
    }

    auto start = std::chrono::high_resolution_clock::now();

    int block_size = 256;
    
    for (size_t i = 0; i < N; i += chunk_size) {
        size_t current_chunk = min(chunk_size, N - i);
        int stream_idx = i / chunk_size;
        
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_a + i, h_a + i, 
                                  current_chunk * sizeof(int),
                                  cudaMemcpyHostToDevice, streams[stream_idx]));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_b + i, h_b + i, 
                                  current_chunk * sizeof(int),
                                  cudaMemcpyHostToDevice, streams[stream_idx]));
        
        int grid_size = (current_chunk + block_size - 1) / block_size;
        vector_addition_kernel<<<grid_size, block_size, 0, streams[stream_idx]>>>(d_a + i, d_b + i, d_c + i, current_chunk);
        
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_c + i, d_c + i, 
                                  current_chunk * sizeof(int),
                                  cudaMemcpyDeviceToHost, streams[stream_idx]));
    }

    for (auto& stream : streams) {
        CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Chunk size: %zu KB, (%zu elem)\n", chunk_size / 1024, chunk_size);
    printf("Time: %.4f ms\n\n", elapsed.count());

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK_RETURN(cudaFree(d_a));
    CUDA_CHECK_RETURN(cudaFree(d_b));
    CUDA_CHECK_RETURN(cudaFree(d_c));
    for (auto& stream : streams) {
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream));
    }
}

int main() {
    srand(time(nullptr));
    const size_t N = 1 << 24;
    printf("Размер вектора: %zu\n", N);

    std::vector<size_t> chunk_sizes = {
        1 << 10,      
        1 << 12,    
        1 << 14,    
        1 << 16,    
        1 << 18,    
        1 << 20,    
        1 << 22,
        1 << 24,
    };

    for (size_t chunk : chunk_sizes) {
        benchmark_with_chunk_size(N, chunk);
    }

    return 0;
}