#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <chrono>

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}


#define M 4096
#define N 4096
#define K 4096

#define LDA M
#define LDB K
#define LDC M

void mm_no_tensor_cores(float *A,  float *B, float *C) {
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    cublasSgemm(
        cublas_handle, 
        CUBLAS_OP_T, 
        CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, LDA,
        B, LDB,
        &beta, 
        C, LDC 
    );

    cublasDestroy(cublas_handle);
}

void mm_tensor_cores(float *A, float *B, float *C) {
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    cublasSgemm(
        cublas_handle, 
        CUBLAS_OP_T, 
        CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, LDA,
        B, LDB,
        &beta, 
        C, LDC 
    );

    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    cublasDestroy(cublas_handle);
}

void fill_random(float *vec, int size) {
    for (int i = 0; i < size; i++)
        vec[i] = (float) rand() / RAND_MAX;
}

void process_mm() {

    float *h_A, *h_B, *h_C_no_tensors, *h_C_tensors;
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C_no_tensors = new float[M * N];
    h_C_tensors = new float[M * N];

    fill_random(h_A, M * K);
    fill_random(h_A, K * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK_RETURN(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));


    auto start = std::chrono::high_resolution_clock::now();
    mm_no_tensor_cores(d_A, d_B, d_C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> no_tensors_time = end - start;
    CUDA_CHECK_RETURN(cudaMemcpy(h_C_no_tensors, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    start = std::chrono::high_resolution_clock::now();
    mm_tensor_cores(d_A, d_B, d_C);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> tensors_time = end - start;
    CUDA_CHECK_RETURN(cudaMemcpy(h_C_tensors, d_C, M * N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("Время выполнения cuBLAS (без тензорных ядер): %.4f мс\n", no_tensors_time.count());
    printf("Время выполнения cuBLAS (с тензорными ядрами): %.4f мс\n", tensors_time.count());

}

int main(int argc, char *argv[]) {
    process_mm();
    return 0;
}