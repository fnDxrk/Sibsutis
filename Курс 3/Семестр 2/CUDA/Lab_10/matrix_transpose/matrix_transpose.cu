#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>

#include <iostream>
#include <chrono>

#define TILE_SIZE 32

__host__ double random_double() {
    return ((double) rand() / RAND_MAX);
}

__host__ void get_transpose_indexes(thrust::host_vector<int> &map, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        map[i] = (i % cols) * rows + (i / cols); 
}

// __global__ void transpose_kernel(double *d_in, double *d_out, int rows, int cols) {

//     __shared__ double tile[TILE_SIZE][TILE_SIZE];

//     int x_in = blockIdx.x * TILE_SIZE + threadIdx.x;
//     int y_in = blockIdx.y * TILE_SIZE + threadIdx.y;

//     if (x_in < cols && y_in < rows)
//         tile[threadIdx.y][threadIdx.x] = d_in[y_in * cols + x_in];

//     __syncthreads();

//     int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
//     int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;
    
//     if (x_out < rows && y_out < cols)
//         d_out[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y]; 
// }

__global__ void transpose_kernel(double *d_in, double *d_out, uint width, uint height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        d_out[x * height + y] = d_in[y * width + x];
    }
}


void thrust_matrix_transponiton(
    thrust::device_vector<double> &d_matrix,
    thrust::device_vector<double> &d_tmatrix,
    thrust::device_vector<int> &d_map
) {

    thrust::gather(
        d_map.begin(),
        d_map.end(),
        d_matrix.begin(),
        d_tmatrix.begin()
    );
}

void time_test(int size) {
    
    int rows = size;
    int cols = size;
    size_t size_bytes = rows * cols * sizeof(double);

    thrust::host_vector<double> h_matrix(rows * cols);
    std::generate(h_matrix.begin(), h_matrix.end(), random_double);

    double *cuda_d_matrix, *cuda_d_tmatrix;
    cudaMalloc(&cuda_d_matrix, size_bytes);
    cudaMalloc(&cuda_d_tmatrix, size_bytes);
    cudaMemcpy(cuda_d_matrix, h_matrix.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks(
        (cols + TILE_SIZE - 1) / TILE_SIZE,
        (rows + TILE_SIZE - 1) / TILE_SIZE
    );
    
    thrust::host_vector<int> h_map(rows * cols);
    get_transpose_indexes(h_map, rows, cols);
    
    thrust::device_vector<double> thrust_d_matrix = h_matrix;
    thrust::device_vector<double> thrust_d_tmatrix(thrust_d_matrix.size());
    thrust::device_vector<int> thrust_d_map = h_map;

    auto cuda_start = std::chrono::high_resolution_clock::now();
    transpose_kernel<<<num_blocks, threads_per_block>>>(
        cuda_d_matrix, 
        cuda_d_tmatrix, 
        rows, cols
    );
    cudaDeviceSynchronize();
    auto cuda_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cuda_time = cuda_end - cuda_start;

    auto thrust_start = std::chrono::high_resolution_clock::now();
    thrust_matrix_transponiton(thrust_d_matrix, thrust_d_tmatrix, thrust_d_map);
    auto thrust_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> thrust_time = thrust_end - thrust_start;
    
    printf("Размер матриц: %d x %d\n", rows, cols);
    printf("Время выполнения (CUDA API): %.4f мс\n", cuda_time.count());
    printf("Время выполнения (Thrust): %.4f мс\n\n", thrust_time.count());
}

int main(int argc, char *argv[]) {
    srand(time(nullptr));

    time_test(1 << 8);     // 256
    time_test(1 << 10);    // 1024
    time_test(1 << 11);    // 2048
    time_test(1 << 12);    // 4096
    time_test(1 << 13);    // 8192
    
    return 0;
}