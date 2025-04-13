extern "C" __global__ void matrix_multiplication(float *vec1, float *vec2, float *res, size_t size) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; i++) {
            sum += vec1[row * size + i] * vec2[i * size + col];
        }
        res[row * size + col] = sum;
    }
}