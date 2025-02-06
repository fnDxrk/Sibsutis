import numpy as np

def half_fast_fourier_transform(f):
    N = len(f)
    p1 = int(np.sqrt(N))
    p2 = N // p1

    assert p1 * p2 == N, "N должно быть произведением p1 и p2"

    A_1 = np.zeros((p1, p2), dtype=complex)
    for k2 in range(p2):
        for k1 in range(p1):
            summation = 0
            for j1 in range(p1):
                j = j1 + p1 * k2
                exponent = -2j * np.pi * k1 * j1 / p1
                summation += f[j] * np.exp(exponent)
            A_1[k1, k2] = summation

    A_2 = np.zeros((p1, p2), dtype=complex)
    for k1 in range(p1):
        for k2 in range(p2):
            summation = 0
            for j2 in range(p2):
                j = k1 + p1 * j2
                exponent = -2j * np.pi * k2 * j2 / p2
                summation += A_1[k1, j2] * np.exp(exponent)
            A_2[k1, k2] = summation

    A = np.zeros(N, dtype=complex)
    for k1 in range(p1):
        for k2 in range(p2):
            k = k1 + p1 * k2
            A[k] = A_2[k1, k2]

    return A

def inverse_half_fast_fourier_transform(F):
    N = len(F)
    p1 = int(np.sqrt(N))
    p2 = N // p1

    assert p1 * p2 == N, "N должно быть произведением p1 и p2"

    F_reshaped = np.zeros((p1, p2), dtype=complex)
    for k1 in range(p1):
        for k2 in range(p2):
            k = k1 + p1 * k2
            F_reshaped[k1, k2] = F[k]

    A_inv_1 = np.zeros((p1, p2), dtype=complex)
    for j2 in range(p2):
        for j1 in range(p1):
            summation = 0
            for k2 in range(p2):
                k = j1 + p1 * k2
                exponent = 2j * np.pi * k2 * j2 / p2
                summation += F_reshaped[j1, k2] * np.exp(exponent)
            A_inv_1[j1, j2] = summation

    A_inv_2 = np.zeros((p1, p2), dtype=complex)
    for j1 in range(p1):
        for j2 in range(p2):
            summation = 0
            for k1 in range(p1):
                k = k1 + p1 * j2
                exponent = 2j * np.pi * k1 * j1 / p1
                summation += A_inv_1[k1, j2] * np.exp(exponent)
            A_inv_2[j1, j2] = summation

    A_inv = np.zeros(N, dtype=complex)
    for j1 in range(p1):
        for j2 in range(p2):
            j = j1 + p1 * j2
            A_inv[j] = A_inv_2[j1, j2] / N

    return A_inv

f = np.array([1, 2, 3, 4], dtype=complex)

result = half_fast_fourier_transform(f)
inverse_result = inverse_half_fast_fourier_transform(result)

fft_result = np.fft.fft(f)
ifft_result = np.fft.ifft(fft_result)

print("Результат полубыстрого преобразования Фурье (ПШФ):")
print(result)

print("\nРезультат обратного полубыстрого преобразования Фурье:")
print(inverse_result)

print("\nРезультат обычного FFT для сравнения:")
print(fft_result)
