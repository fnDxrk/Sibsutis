#include <cuda_runtime.h>
#include <iostream>
#include <chrono>


void bench(size_t size) {
    float *host_vector = (float*)malloc(size);
    float *pinned_vector;
    float *device_memory;

    cudaMallocHost(&pinned_vector, size);
    cudaMalloc(&device_memory, size);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(device_memory, host_vector, size, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    auto normal_h2d = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(device_memory, pinned_vector, size, cudaMemcpyHostToDevice);
    end = std::chrono::high_resolution_clock::now();
    auto pinned_h2d = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(host_vector, device_memory, size, cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    auto normal_d2h = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(pinned_vector, device_memory, size, cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    auto pinned_d2h = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    std::cout << "Data size: " << size << "\n";
    std::cout << "Host->Device:\n";
    std::cout << "  Normal: " << normal_h2d << " μs\n";
    std::cout << "  Pinned: " << pinned_h2d << " μs\n";
    
    std::cout << "Device->Host:\n";
    std::cout << "  Normal: " << normal_d2h << " μs\n";
    std::cout << "  Pinned: " << pinned_d2h << " μs\n\n";

    free(host_vector);
    cudaFreeHost(pinned_vector);
    cudaFree(device_memory);
}

int main(int argc, char *argv[]) {

    size_t sizes[] = {
        1 << 10,
        1 << 16,
        1 << 20,
        1 << 24,
        1 << 26,
        100'000'000
    };

    for (size_t size : sizes)
        bench(size);


    return 0;
}