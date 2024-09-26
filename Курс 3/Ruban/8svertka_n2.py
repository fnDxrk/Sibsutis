import numpy as np

def DFT(x):
    n = len(x)
    X = np.zeros(n, dtype=complex)
    for k in range(n):
        for t in range(n):
            X[k] += x[t] * np.exp(-2j * np.pi * k * t / n)
    return X

def IDFT(X):
    n = len(X)
    x = np.zeros(n, dtype=complex)
    for t in range(n):
        for k in range(n):
            x[t] += X[k] * np.exp(2j * np.pi * k * t / n)
        x[t] /= n
    return x

def convolution_dft(signal, kernel):
    n = len(signal) + len(kernel) - 1
    
    signal = np.pad(signal, (0, n - len(signal)), mode='constant')
    kernel = np.pad(kernel, (0, n - len(kernel)), mode='constant')
    
    # Применяем ДПФ к сигналу и ядру
    signal_dft = DFT(signal)
    kernel_dft = DFT(kernel)
    
    result_dft = signal_dft * kernel_dft
    
    result = IDFT(result_dft)
    
    return np.real(result)

signal = [1, 2, 3, 4, 5]
kernel = [1, 0, -1]

result = convolution_dft(signal, kernel)

print("Результат свёртки через ДПФ:", result)

# Дополнение нулями:
# Мы дополняем сигнал и ядро нулями до длины n = len(signal) + len(kernel) - 1, чтобы результат свёртки был корректной длины.

# Перемножение в частотной области:
# После преобразования сигналов в частотную область их спектры перемножаются поэлементно.

# Обратное ДПФ:
# После перемножения спектров, мы применяем обратное ДПФ, чтобы получить результат в пространстве времени.

# Пример:
# Если взять сигнал [1, 2, 3, 4, 5] и ядро [1, 0, -1], результат свёртки через ДПФ будет такой же, как если бы мы выполняли обычную свёртку напрямую.
