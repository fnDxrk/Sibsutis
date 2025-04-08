#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

// Объявляем ядро
extern "C" __global__ void kernel(double* vector1, double* vector2, double* result, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
        result[idx] = vector1[idx] + vector2[idx];
}

int main()
{
    int driverVersion;
    cudaDriverGetVersion(&driverVersion);
    std::cout << "CUDA Driver Version: " << driverVersion << std::endl;
    return 0;
}
