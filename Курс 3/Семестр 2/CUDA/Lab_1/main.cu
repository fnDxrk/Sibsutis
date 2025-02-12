#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_cuda()
{
    printf("Hello from CUDA kernel!\n");
}

int main()
{
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
