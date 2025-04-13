#include <cuda.h>
#include <iostream>

#define SIZE 8096

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
    for (size_t i = 0; i < size; i++) {
        vec[i] = rand() / RAND_MAX;
    }
}

void process_matrix_miltiplication() {
    srand(time(nullptr));
    CUDA_CHECK_RETURN(cuInit(0));

    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    
    const char* ptx_file = 
        "/home/dxrk_/Documents/Sibsutis"
        "/Курс 3/Семестр 2/CUDA/Lab_9"
        "/cuda_driver_api/ptx/kernel.ptx";

    CUDA_CHECK_RETURN(cuDeviceGet(&device, 0));
    CUDA_CHECK_RETURN(cuCtxCreate_v2(&context, 0, device));
    CUDA_CHECK_RETURN(cuModuleLoad(&module,ptx_file));
    CUDA_CHECK_RETURN(cuModuleGetFunction(
        &kernel, 
        module, 
        "matrix_multiplication"
    ));

    float *h_vec1, *h_vec2, *h_res;
    h_vec1 = new float[SIZE * SIZE];
    h_vec2 = new float[SIZE * SIZE];
    h_res = new float[SIZE * SIZE];

    fill_random(h_vec1, SIZE * SIZE);
    fill_random(h_vec2, SIZE * SIZE);
    
    CUdeviceptr_v2 d_vec1, d_vec2, d_res;
    CUDA_CHECK_RETURN(cuMemAlloc_v2(
        &d_vec1, 
        SIZE * SIZE * sizeof(float)
    ));
    CUDA_CHECK_RETURN(cuMemAlloc_v2(
        &d_vec2,
        SIZE * SIZE * sizeof(float)
    ));
    CUDA_CHECK_RETURN(cuMemAlloc_v2(
        &d_res,
        SIZE * SIZE * sizeof(float)
    ));

    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (SIZE + threads_per_block.x - 1) / threads_per_block.x,
        (SIZE + threads_per_block.y - 1) / threads_per_block.y
    );

    CUevent start, end;
    CUDA_CHECK_RETURN(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CUDA_CHECK_RETURN(cuEventCreate(&end, CU_EVENT_DEFAULT));


    size_t size = SIZE;
    void *args[] = {&d_vec1, &d_vec2, &d_res, &size};
    
    CUDA_CHECK_RETURN(cuEventRecord(start, 0));
    CUDA_CHECK_RETURN(cuLaunchKernel(
        kernel,
        num_blocks.x, num_blocks.y, num_blocks.z,
        threads_per_block.x, threads_per_block.y, threads_per_block.z,
        0,            // sharedMemBytes (0 по умолчанию)
        nullptr,      // stream (nullptr для дефолтного)
        args,         // массив указателей на аргументы
        nullptr       // дополнительные параметры
    ));
    CUDA_CHECK_RETURN(cuEventRecord(end, 0));
    CUDA_CHECK_RETURN(cuEventSynchronize(end));
    CUDA_CHECK_RETURN(cuCtxSynchronize());

    CUDA_CHECK_RETURN(cuMemcpyDtoH_v2(
        h_res, 
        d_res, 
        SIZE * SIZE * sizeof(float)
    ));

    float ellapsed_time;
    CUDA_CHECK_RETURN(cuEventElapsedTime(&ellapsed_time, start, end));
    printf("Время выполнения: %.4f мс\n", ellapsed_time);

    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_res;

    CUDA_CHECK_RETURN(cuMemFree_v2(d_vec1));
    CUDA_CHECK_RETURN(cuMemFree_v2(d_vec2));
    CUDA_CHECK_RETURN(cuMemFree_v2(d_res));

    CUDA_CHECK_RETURN(cuEventDestroy_v2(start));
    CUDA_CHECK_RETURN(cuEventDestroy_v2(end));

    CUDA_CHECK_RETURN(cuModuleUnload(module));
    CUDA_CHECK_RETURN(cuCtxDestroy_v2(context));

}

int main(int argc, char *argv[]) {
    process_matrix_miltiplication();
    return 0;
}
