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

f = np.array([1, 2, 3, 4], dtype=complex)

result = half_fast_fourier_transform(f)

fft_result = np.fft.fft(f)

print("Результат полубыстрого преобразования Фурье (ПШФ):")
print(result)

print("\nРезультат обычного FFT для сравнения:")
print(fft_result)
