import numpy as np

from numba import cuda

@cuda.jit
def matrix_multiplication(vec1, vec2, res, size):    
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if (row < size and col < size):
        sum = 0.0
        for i in range(size):
            sum += vec1[row * size + i] * vec2[i * size + col]
        res[row * size + col] = sum


def process_matrix_multiplication():
    SIZE = 8096
    # Создание векторов на хосте
    h_vec1 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_vec2 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_res = np.zeros(SIZE * SIZE, dtype=np.float32)

    # Копирование данных на устройство
    d_vec1 = cuda.to_device(h_vec1)
    d_vec2 = cuda.to_device(h_vec2)
    d_res = cuda.device_array_like(h_res)

    # Конфигурация для запуска ядра
    threads_per_block = (16, 16)
    blocks_per_grid_x = (SIZE + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (SIZE + threads_per_block[1] - 1) // threads_per_block[1]
    num_blocks = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Регистрация событий для подсчета времени 
    start_event = cuda.event()
    end_event = cuda.event()

    
    start_event.record()
    matrix_multiplication[num_blocks, threads_per_block](d_vec1, d_vec2, d_res, SIZE)
    end_event.record()
    cuda.synchronize()

    # Копирование данных на хост
    h_res = d_res.copy_to_host()

    # Подсчет времени выполнения
    ellapsed_time = cuda.event_elapsed_time(start_event, end_event)
    print(f"Время выполнения: {ellapsed_time:.4f} мс")



if __name__ == "__main__":
    process_matrix_multiplication()
    pass