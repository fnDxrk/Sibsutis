import numpy as np

def half_fast_fourier_transform(f):
    N = len(f)

    p1 = int(np.sqrt(N))
    p2 = N // p1

    assert p1 * p2 == N

    A = np.zeros(N, dtype=complex)

    for k1 in range(p1):
        for k2 in range(p2):
            summation = 0
            for j1 in range(p1):
                for j2 in range(p2):
                    j = j1 + p1 * j2
                    k = k1 + p1 * k2

                    exponent = -2j * np.pi * (k1 * j1 / p1 + k2 * j2 / p2)
                    summation += f[j] * np.exp(exponent)

            A[k1 + p1 * k2] = summation
    
    return A

def inverse_half_fast_fourier_transform(F):
    N = len(F)

    p1 = int(np.sqrt(N))
    p2 = N // p1

    assert p1 * p2 == N

    A_inv = np.zeros(N, dtype=complex)

    for j1 in range(p1):
        for j2 in range(p2):
            summation = 0
            for k1 in range(p1):
                for k2 in range(p2):
                    j = j1 + p1 * j2
                    k = k1 + p1 * k2

                    exponent = 2j * np.pi * (k1 * j1 / p1 + k2 * j2 / p2)
                    summation += F[k] * np.exp(exponent)

            A_inv[j1 + p1 * j2] = summation / N
    
    return A_inv

# Тестирование
f = np.array([1, 2, 3, 4], dtype=complex)

# Прямое полубыстрое преобразование Фурье
result = half_fast_fourier_transform(f)

# Обратное полубыстрое преобразование Фурье
inverse_result = inverse_half_fast_fourier_transform(result)

# Обычные FFT и обратное FFT для сравнения
fft_result = np.fft.fft(f)
ifft_result = np.fft.ifft(fft_result)

print("Результат полубыстрого преобразования Фурье (ПШФ):")
print(result)

print("\nРезультат обратного полубыстрого преобразования Фурье:")
print(inverse_result)

print("\nРезультат обычного FFT для сравнения:")
print(fft_result)

print("\nРезультат обратного FFT для сравнения:")
print(ifft_result)

