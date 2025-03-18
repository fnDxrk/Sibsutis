#include <cuda_runtime.h>
#include <stdio.h>

#define N 4   // Количество векторов
#define K 3   // Длина каждого вектора

// Ядро копирования без эмуляции register pressure
__global__ void copyKernel(const float *a, float *b, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;
    if(idx < total)
    {
        // Определяем номер вектора и индекс элемента внутри вектора
        int i = idx / k;   // номер вектора (0 ... n-1)
        int j = idx % k;   // позиция элемента внутри вектора (0 ... k-1)
        // Новая позиция в массиве b: элементы группируются по j
        int newIdx = j * n + i;
        b[newIdx] = a[idx];
    }
}

// Ядро с эмуляцией высокого register pressure.
__global__ void copyKernelWithPressure(const float *a, float *b, int n, int k)
{
    // Большой локальный массив для эмуляции register pressure
    __shared__ float dummy[1024]; // Например, можно использовать shared память как эмуляцию
    // Или можно создать большой массив в регистровой памяти
    float pressure[256];
    for (int i = 0; i < 256; i++) {
        pressure[i] = 0.0f;
    }
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;
    if(idx < total)
    {
        int i = idx / k;
        int j = idx % k;
        int newIdx = j * n + i;
        pressure[threadIdx.x % 256] = a[idx];
        b[newIdx] = pressure[threadIdx.x % 256];
    }
}

int main(void)
{
    const int totalElements = N * K;
    const int size = totalElements * sizeof(float);

    // Выделяем память на хосте
    float h_a[totalElements], h_b[totalElements];
    for (int i = 0; i < totalElements; i++) {
        h_a[i] = (float)i;
    }

    // Выделяем память на устройстве
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    // Копируем данные с хоста на устройство
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Определяем параметры запуска ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск ядра копирования без register pressure
    //copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N, K);
    copyKernelWithPressure<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N, K);

    // Ожидание завершения всех потоков на устройстве
    cudaDeviceSynchronize();

    // Копируем результат обратно на хост
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    // Вывод результата для проверки
    printf("Массив a:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%4.0f ", h_a[i * K + j]);
        }
        printf("\n");
    }

    printf("\nМассив b:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4.0f ", h_b[i * N + j]);
        }
        printf("\n");
    }

    // Освобождаем память на устройстве
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}