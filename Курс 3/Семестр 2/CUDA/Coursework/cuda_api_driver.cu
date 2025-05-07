#include <cuda.h>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

#define CUDA_CHECK_RETURN(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CUDA error: " << errStr << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

void fill_random(float *vec, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
}

void process_matrix_multiplication(size_t size) {
    CUDA_CHECK_RETURN(cuInit(0));

    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    const char* ptx_file = "./kernel.ptx";

    CUDA_CHECK_RETURN(cuDeviceGet(&device, 0));
    CUDA_CHECK_RETURN(cuCtxCreate(&context, 0, device));
    CUDA_CHECK_RETURN(cuModuleLoad(&module, ptx_file));
    CUDA_CHECK_RETURN(cuModuleGetFunction(&kernel, module, "matrix_multiplication"));

    float *h_vec1, *h_vec2, *h_res;
    h_vec1 = new float[size * size];
    h_vec2 = new float[size * size];
    h_res = new float[size * size];

    fill_random(h_vec1, size * size);
    fill_random(h_vec2, size * size);

    CUdeviceptr d_vec1, d_vec2, d_res;
    CUDA_CHECK_RETURN(cuMemAlloc(&d_vec1, size * size * sizeof(float)));
    CUDA_CHECK_RETURN(cuMemAlloc(&d_vec2, size * size * sizeof(float)));
    CUDA_CHECK_RETURN(cuMemAlloc(&d_res, size * size * sizeof(float)));

    CUDA_CHECK_RETURN(cuMemcpyHtoD(d_vec1, h_vec1, size * size * sizeof(float)));
    CUDA_CHECK_RETURN(cuMemcpyHtoD(d_vec2, h_vec2, size * size * sizeof(float)));

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((size + threads_per_block.x - 1) / threads_per_block.x,
                    (size + threads_per_block.y - 1) / threads_per_block.y);

    CUevent start, end;
    CUDA_CHECK_RETURN(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CUDA_CHECK_RETURN(cuEventCreate(&end, CU_EVENT_DEFAULT));

    size_t kernel_size = size;
    void *args[] = {&d_vec1, &d_vec2, &d_res, &kernel_size};

    const int iterations = 10;
    float total_time = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK_RETURN(cuEventRecord(start, 0));
        CUDA_CHECK_RETURN(cuLaunchKernel(
            kernel,
            num_blocks.x, num_blocks.y, 1,
            threads_per_block.x, threads_per_block.y, 1,
            0, nullptr, args, nullptr));
        CUDA_CHECK_RETURN(cuEventRecord(end, 0));
        CUDA_CHECK_RETURN(cuEventSynchronize(end));
        float iter_time;
        CUDA_CHECK_RETURN(cuEventElapsedTime(&iter_time, start, end));
        total_time += iter_time;
    }
    CUDA_CHECK_RETURN(cuCtxSynchronize());

    CUDA_CHECK_RETURN(cuMemcpyDtoH(h_res, d_res, size * size * sizeof(float)));

    printf("Driver API, Matrix Size %lux%lu: %.4f мс (усреднено за %d итераций)\n",
           size, size, total_time / iterations, iterations);

    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_res;

    CUDA_CHECK_RETURN(cuMemFree(d_vec1));
    CUDA_CHECK_RETURN(cuMemFree(d_vec2));
    CUDA_CHECK_RETURN(cuMemFree(d_res));

    CUDA_CHECK_RETURN(cuEventDestroy(start));
    CUDA_CHECK_RETURN(cuEventDestroy(end));
    CUDA_CHECK_RETURN(cuModuleUnload(module));
    CUDA_CHECK_RETURN(cuCtxDestroy(context));
}

int main() {
    std::vector<size_t> sizes = {256, 512, 1024, 2048, 4096, 8192};
    for (size_t size : sizes) {
        process_matrix_multiplication(size);
    }
    return 0;
}