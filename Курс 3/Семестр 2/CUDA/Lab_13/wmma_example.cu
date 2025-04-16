#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>


#define M 4096
#define N 4096
#define K 4096

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

void fill_random(half *vec, int size) {
    for (int i = 0; i < size; i++) {
        float random_value = (float) rand() / RAND_MAX;
        vec[i] = __float2half(random_value);
    }
}

__global__ void wmma_gemm_kernel(half *a, half *b, half *c, int m, int n, int k) {
    using namespace nvcuda;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warpM * WMMA_M >= m || warpN * WMMA_N >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        const half *tile_a = a + aRow * k + aCol;
        const half *tile_b = b + bRow * n + bCol;

        wmma::load_matrix_sync(a_frag, tile_a, k);
        wmma::load_matrix_sync(b_frag, tile_b, n);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    half *tile_c = c + warpM * WMMA_M * n + warpN * WMMA_N;
    wmma::store_matrix_sync(tile_c, c_frag, n, wmma::mem_row_major);
}

void run_wmma() {
    half *h_A, *h_B, *h_C;

    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C = new half[M * N];

    fill_random(h_A, M * K);
    fill_random(h_B, K * N);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemset(d_C, 0, M * N * sizeof(half));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);  // Оптимизируем размер блока для эффективного использования SM
    dim3 numBlocks((M + (WMMA_M * 4 - 1)) / (WMMA_M * 4), (N + WMMA_N - 1) / WMMA_N);

    auto start = std::chrono::high_resolution_clock::now();
    wmma_gemm_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Время выполнения WMMA: " << duration.count() << " мс" << std::endl;

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    run_wmma();
    return 0;
}
