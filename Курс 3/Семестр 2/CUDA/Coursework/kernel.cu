extern "C" __global__ void matrix_multiplication(float *vec1, float *vec2, float *res, size_t size) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (size_t i = 0; i < size; i += 16) {
        if (row < size && i + threadIdx.x < size)
            sA[threadIdx.y][threadIdx.x] = vec1[row * size + i + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < size && i + threadIdx.y < size)
            sB[threadIdx.y][threadIdx.x] = vec2[(i + threadIdx.y) * size + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (size_t k = 0; k < 16; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < size && col < size)
        res[row * size + col] = sum;
}