#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <pthread.h>
#include <vector>

void add_vectors_seq(const float* a, const float* b, float* c, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

struct ThreadData {
    float* a;
    float* b;
    float* c;
    size_t start, end;
};

void* add_vectors_parallel(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    for (size_t i = data->start; i < data->end; ++i) {
        data->c[i] = data->a[i] + data->b[i];
    }
    return nullptr;
}

void add_vectors_pthreads(float* a, float* b, float* c, size_t n, size_t num_threads)
{
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);

    size_t chunk_size = n / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        thread_data[i] = { a, b, c, i * chunk_size, (i == num_threads - 1) ? n : (i + 1) * chunk_size };
        pthread_create(&threads[i], nullptr, add_vectors_parallel, &thread_data[i]);
    }

    for (pthread_t& t : threads) {
        pthread_join(t, nullptr);
    }
}

__global__ void add_vectors_cuda(const float* a, const float* b, float* c, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void add_vectors_gpu(float* a, float* b, float* c, size_t n)
{
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_vectors_cuda<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    size_t n = 1000000;

    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    add_vectors_seq(a.data(), b.data(), c.data(), n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seq_duration = end - start;
    std::cout << "Sequential time: " << seq_duration.count() << " seconds\n";

    size_t num_threads = 4;
    std::fill(c.begin(), c.end(), 0.0f);
    start = std::chrono::high_resolution_clock::now();
    add_vectors_pthreads(a.data(), b.data(), c.data(), n, num_threads);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pthreads_duration = end - start;
    std::cout << "Pthreads time (threads = " << num_threads << "): " << pthreads_duration.count() << " seconds\n";

    std::fill(c.begin(), c.end(), 0.0f);
    start = std::chrono::high_resolution_clock::now();
    add_vectors_gpu(a.data(), b.data(), c.data(), n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end - start;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds\n";

    return 0;
}
