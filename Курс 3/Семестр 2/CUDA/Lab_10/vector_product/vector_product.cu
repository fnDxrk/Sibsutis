#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <iostream>
#include <chrono>


void fill_random(double *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (double) rand() / RAND_MAX;
    }
}

__global__ void scalar_product_kernel(double *vec1, double *vec2, double *res, int size) {

	extern __shared__ double cache[];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cache_index = threadIdx.x;
	
	double temp = 0;
	while (tid < size){
		temp += vec1[tid] * vec2[tid];
		tid += blockDim.x * gridDim.x;
	}
	
	cache[cache_index] = temp;
	__syncthreads();
	
	int i = blockDim.x / 2;
	while (i != 0){
		if (cache_index < i)
			cache[cache_index] += cache[cache_index + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cache_index == 0)
		res[blockIdx.x] = cache[0];
}

void cuda_scalar_product(
    double *d_vec1, 
    double *d_vec2, 
    double *d_res,
    double *res, 
    int size,
    dim3 threads_per_block,
    int num_blocks
) {
    
    scalar_product_kernel<<<num_blocks, threads_per_block, threads_per_block.x * sizeof(double)>>>(d_vec1, d_vec2, d_res, size);
    cudaDeviceSynchronize();

    double *partial_sums = new double[num_blocks];
    cudaMemcpy(
        partial_sums,
        d_res,
        num_blocks * sizeof(double),
        cudaMemcpyDeviceToHost
    );

    *res = 0;
    for (int i = 0; i < num_blocks; i++)
        *res += partial_sums[i];


    delete[] partial_sums;
}

void thrust_scalar_product_v1(
    thrust::device_vector<double> &d_vec1, 
    thrust::device_vector<double> &d_vec2,
    double *res
    
) {

    *res = thrust::inner_product(
        d_vec1.begin(),
        d_vec1.end(),
        d_vec2.begin(),
        0.0
    );

}

void thrust_scalar_product_v2(
    thrust::device_vector<double> &d_vec1,
    thrust::device_vector<double> &d_vec2,
    double *res

) {

    thrust::device_vector<double> multiply_res;

    thrust::transform(
        d_vec1.begin(),
        d_vec1.end(),
        d_vec2.begin(),
        multiply_res.begin(),
        thrust::multiplies<double>()
    );

    *res = thrust::reduce(
        multiply_res.begin(),
        multiply_res.end(),
        0.0,
        thrust::plus<double>()
    );
}


void time_test(int size) {
    
    double cuda_sp_res;
    double thrust_sp_res;

    double *vec1, *vec2;
    vec1 = new double[size];
    vec2 = new double[size];
    fill_random(vec1, size);
    fill_random(vec2, size);
    
    dim3 threads_per_block(256);
    int num_blocks = (size + threads_per_block.x - 1) / threads_per_block.x;

    double *cuda_d_vec1, *cuda_d_vec2, *cuda_d_res;
    cudaMalloc(&cuda_d_vec1, size * sizeof(double));
    cudaMalloc(&cuda_d_vec2, size * sizeof(double));
    cudaMalloc(&cuda_d_res, num_blocks * sizeof(double));
    cudaMemcpy(cuda_d_vec1, vec1, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_d_vec2, vec2, size * sizeof(double), cudaMemcpyHostToDevice);

    thrust::device_vector<double> thrust_d_vec1(vec1, vec1 + size);
    thrust::device_vector<double> thrust_d_vec2(vec2, vec2 + size);
    
    auto cuda_start = std::chrono::high_resolution_clock::now();
    cuda_scalar_product(
        cuda_d_vec1, 
        cuda_d_vec2, 
        cuda_d_res, 
        &cuda_sp_res, 
        size, 
        threads_per_block, 
        num_blocks
    );
    auto cuda_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cuda_time = cuda_end - cuda_start;

    auto thrust_start = std::chrono::high_resolution_clock::now();
    thrust_scalar_product_v1(thrust_d_vec1, thrust_d_vec2, &thrust_sp_res);
    auto thrust_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> thrust_time = thrust_end - thrust_start; 
    
    printf("Размер векторов: %d\n", size);
    printf("Время выполнения (CUDA API): %.4f мс\n", cuda_time.count());
    printf("Время выполнения (Thrust): %.4f мс\n", thrust_time.count());
    printf("Результат вычислений (CUDA API): %4.f\n", cuda_sp_res);
    printf("Результат вычислений (Thrust): %4.f\n\n", thrust_sp_res);

    delete[] vec1;
    delete[] vec2;

    cudaFree(cuda_d_vec1);
    cudaFree(cuda_d_vec2);
    cudaFree(cuda_d_res);
}

int main(int argc, char *argv[]) {
    time_test(1 << 8);     // 256
    time_test(1 << 10);    // 1024
    
    time_test(1 << 12);    // 4096
    time_test(1 << 14);    // 16384
    time_test(1 << 16);    // 65536
    
    time_test(1 << 18);    // 262144
    time_test(1 << 20);    // 1M
    time_test(1 << 22);    // 4M
    time_test(1 << 24);    // 16M
    time_test(1 << 26);    // 64M

    time_test(100'000'000);

    return 0;
}