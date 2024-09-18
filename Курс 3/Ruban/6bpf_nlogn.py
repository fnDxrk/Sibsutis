import numpy as np

def fft_1d(x):
    """
    Реализует Быстрое Преобразование Фурье (FFT) для одномерного массива x.
    """
    N = len(x)
    if N <= 1:
        return x
    
    # Декомпозиция на четные и нечетные индексы
    even = fft_1d(x[::2])
    odd = fft_1d(x[1::2])
    
    # Расчет комплексных экспонент
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # Объединение результатов
    result = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        result[k] = even[k] + factor[k] * odd[k]
        result[k + N // 2] = even[k] - factor[k] * odd[k]
    
    return result

# Пример одномерного массива
x = np.array([1, 2, 3, 4], dtype=complex)

# Вычисление FFT без библиотеки
fft_result_custom = fft_1d(x)

# Вычисление FFT с использованием библиотеки numpy
fft_result_numpy = np.fft.fft(x)

print("FFT result (custom implementation, 1D):")
print(fft_result_custom)

print("\nFFT result (numpy implementation, 1D):")
print(fft_result_numpy)

# Проверка близости результатов
print("\nAre the results close?")
print(np.allclose(fft_result_custom, fft_result_numpy))
