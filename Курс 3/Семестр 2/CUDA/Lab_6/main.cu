#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SH_DIM 32

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK_RETURN(value) {                                   \
    cudaError_t _m_cudaStat = value;                                 \
    if (_m_cudaStat != cudaSuccess) {                                \
        fprintf(stderr, "Error %s at line %d in file %s\n",           \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1);                                                     \
    } }

// Инициализация матрицы (каждый элемент = float(k + n*K))
__global__ void gInit(float* a) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    int K = blockDim.x * gridDim.x;
    a[k + n * K] = (float)(k + n * K);
}

// Ядро транспонирования без разделяемой памяти
__global__ void gTranspose1(float* a, float* b) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    int K = blockDim.x * gridDim.x;
    // Чтение из a и запись в транспонированном виде в b
    b[k + n * K] = a[n + k * K];
}

// Альтернативное ядро без разделяемой памяти (вариант с индексами, меняющими порядок записи)
__global__ void gTranspose2(float* a, float* b) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    int K = blockDim.x * gridDim.x;
    int N = blockDim.y * gridDim.y;
    // Здесь b записывается как b[n + k * N]
    b[n + k * N] = a[k + n * K];
}

// Ядро с использованием разделяемой памяти (без разрешения банковых конфликтов)
__global__ void gTransposeSM(float* a, float* b) {
    __shared__ float cache[SH_DIM][SH_DIM];
    
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;
    
    // Загружаем данные из глобальной памяти в разделяемую память
    cache[threadIdx.y][threadIdx.x] = a[k + n * N];
    __syncthreads();
    
    // Определяем новые индексы для записи результата
    k = threadIdx.x + blockIdx.y * blockDim.x;
    n = threadIdx.y + blockIdx.x * blockDim.y;
    b[k + n * N] = cache[threadIdx.x][threadIdx.y];
}

// Ядро с использованием разделяемой памяти с разрешением конфликтов банков
__global__ void gTransposeSM_WC(float* a, float* b) {
    // Размер второго измерения увеличен на 1 для устранения конфликтов
    __shared__ float cache[SH_DIM][SH_DIM+1];
    
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;
    
    cache[threadIdx.y][threadIdx.x] = a[k + n * N];
    __syncthreads();
    
    k = threadIdx.x + blockIdx.y * blockDim.x;
    n = threadIdx.y + blockIdx.x * blockDim.y;
    b[k + n * N] = cache[threadIdx.x][threadIdx.y];
}

int main(int argc, char* argv[]){
    if(argc < 3){
        fprintf(stderr, "USAGE: %s <dimension of matrix> <dimension of threads>\n", argv[0]);
        return -1;
    }
    
    int N = atoi(argv[1]);
    int dim_of_threads = atoi(argv[2]);
    
    if(N % dim_of_threads){
        fprintf(stderr, "Matrix dimension must be divisible by thread block dimension\n");
        return -1;
    }
    
    int dim_of_blocks = N / dim_of_threads;
    const int max_size = 1 << 8;
    if(dim_of_blocks > max_size){
        fprintf(stderr, "Too many blocks\n");
        return -1;
    }
    
    size_t bytes = N * N * sizeof(float);
    float *a, *b;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&a, bytes));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&b, bytes));
    
    // Задаём размеры сетки и блока
    dim3 threads(dim_of_threads, dim_of_threads);
    dim3 blocks(dim_of_blocks, dim_of_blocks);
    
    // Инициализация матрицы
    gInit<<<blocks, threads>>>(a);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    
    // Очистка памяти для b
    CUDA_CHECK_RETURN(cudaMemset(b, 0, bytes));
    
    // Транспонирование без разделяемой памяти (ядро gTranspose2)
    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));
    
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
    gTranspose2<<<blocks, threads>>>(a, b);
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    
    float elapsed;
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed, start, stop));
    printf("gTranspose2 elapsed time: %f ms\n", elapsed);
    
    // Очистка результата
    CUDA_CHECK_RETURN(cudaMemset(b, 0, bytes));
    
    // Транспонирование с использованием разделяемой памяти (без разрешения конфликтов)
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
    gTransposeSM<<<blocks, threads>>>(a, b);
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed, start, stop));
    printf("gTransposeSM elapsed time: %f ms\n", elapsed);
    
    CUDA_CHECK_RETURN(cudaMemset(b, 0, bytes));
    
    // Транспонирование с использованием разделяемой памяти с разрешением конфликтов банков
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
    gTransposeSM_WC<<<blocks, threads>>>(a, b);
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed, start, stop));
    printf("gTransposeSM_WC elapsed time: %f ms\n", elapsed);
    
    // Освобождение ресурсов
    CUDA_CHECK_RETURN(cudaFree(a));
    CUDA_CHECK_RETURN(cudaFree(b));
    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(stop));
    
    return 0;
}
