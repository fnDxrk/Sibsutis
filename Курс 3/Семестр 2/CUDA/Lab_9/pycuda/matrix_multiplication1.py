import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import gpuarray
from pycuda.compiler import SourceModule

cuda_code = """
    __global__ void matrix_multiplication(float *vec1, float *vec2, float *res, int size) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += vec1[row * size + i] * vec2[i * size + col];
        }
        res[row * size + col] = sum;
    }
}
"""

def process_matrix_multiplication():
    SIZE = 8096
    # Создание векторов на хосте
    h_vec1 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_vec2 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_res = np.zeros(SIZE * SIZE, dtype=np.float32)

    # Копирование данных с хоста на устройство
    d_vec1 = gpuarray.to_gpu(h_vec1)
    d_vec2 = gpuarray.to_gpu(h_vec2)
    d_res = gpuarray.zeros_like(d_vec1)

    # Загрузка кода с ядром в модуль
    module = SourceModule(
        cuda_code,
        options=['-arch=sm_89']
    )

    # Получение ядра из модуля
    matrix_multiplication = module.get_function("matrix_multiplication")

    # Конфигурация запуска ядра
    threads_per_block = (16, 16, 1)
    grid_x = (SIZE + threads_per_block[0] - 1) // threads_per_block[0]
    grid_y = (SIZE + threads_per_block[1] - 1) // threads_per_block[1]
    num_blocks = (grid_x, grid_y, 1)

    # Регистрация событий для подсчета времени
    start = cuda.Event()
    end = cuda.Event()
    
    start.record()
    matrix_multiplication(
        d_vec1,
        d_vec2,
        d_res,
        np.int32(SIZE),
        block=threads_per_block,
        grid=num_blocks
    )
    end.record()
    cuda.Context.synchronize()
    
    # Получение данных
    h_res = d_res.get()

    # Подсчет времени
    print(f"Время вычисления: {start.time_till(end):.4f} мс")


if __name__ == "__main__":
    process_matrix_multiplication()
    
