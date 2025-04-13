#include <cuda_runtime.h>
#include <iostream>

#define SIZE 4096

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}


__global__ void matrix_multiplication(float *vec1, float *vec2, float *res, size_t size) {
    
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; i++) {
            sum += vec1[row * size + i] * vec2[i * size + col];
        }
        res[row * size + col] = sum;
    }
}

void fill_random(float *vec, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] = (float) rand() / RAND_MAX;
    }
}

void process_matrix_multiplication() {
    srand(time(nullptr));

    float *h_vec1, *h_vec2, *h_res;
    h_vec1 = new float[SIZE * SIZE]; 
    h_vec2 = new float[SIZE * SIZE]; 
    h_res = new float[SIZE * SIZE];
    
    fill_random(h_vec1, SIZE * SIZE);
    fill_random(h_vec2, SIZE * SIZE);

    float *d_vec1, *d_vec2, *d_res;
    
    CUDA_CHECK_RETURN(cudaMalloc(
        (void**)&d_vec1,
        SIZE * SIZE * sizeof(float)        
    ));
    CUDA_CHECK_RETURN(cudaMalloc(
        (void**)&d_vec2,
        SIZE * SIZE * sizeof(float)
    ));
    CUDA_CHECK_RETURN(cudaMalloc(
        (void**)&d_res,
        SIZE * SIZE * sizeof(float)
    ));

    CUDA_CHECK_RETURN(cudaMemcpy(
        d_vec1, 
        h_vec1, 
        SIZE * SIZE * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_RETURN(cudaMemcpy(
        d_vec2,
        h_vec2,
        SIZE * SIZE * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    dim3 threads_per_block(16, 16);  
    dim3 num_blocks(
        (SIZE + threads_per_block.x - 1) / threads_per_block.x,
        (SIZE + threads_per_block.y - 1) / threads_per_block.y
    );

    cudaEvent_t start, end;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&end));
    
    CUDA_CHECK_RETURN(cudaEventRecord(start));
    matrix_multiplication<<<num_blocks, threads_per_block>>>(d_vec1, d_vec2, d_res, SIZE);
    CUDA_CHECK_RETURN(cudaEventRecord(end));
    CUDA_CHECK_RETURN(cudaEventSynchronize(end));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(
        h_res,
        d_res,
        SIZE * SIZE * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float ellapsed_time;
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&ellapsed_time, start, end));
    printf("Время выполнения: %.4f мс\n", ellapsed_time);

    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_res;
    
    CUDA_CHECK_RETURN(cudaFree(d_vec1));
    CUDA_CHECK_RETURN(cudaFree(d_vec2));
    CUDA_CHECK_RETURN(cudaFree(d_res));

    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(end));
}

int main(int argc, char *argv[]) {
    process_matrix_multiplication();
    return 0;
}