import numpy as np

def fft(x):
    N = len(x)
    if N <= 1:
        return x

    even = fft(x[::2])
    odd = fft(x[1::2])
    
    T = [np.exp(-2j * np.pi * k / N) * odd[k % (N // 2)] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def ifft(x):
    N = len(x)
    if N <= 1:
        return x

    even = ifft(x[::2])
    odd = ifft(x[1::2])
    
    T = [np.exp(2j * np.pi * k / N) * odd[k % (N // 2)] for k in range(N // 2)]
    return [(even[k] + T[k]) / 2 for k in range(N // 2)] + [(even[k] - T[k]) / 2 for k in range(N // 2)]

def convolve(x, y):
    N = len(x) + len(y) - 1
    N = 1 << (N - 1).bit_length()
    
    X = np.pad(x, (0, N - len(x)), 'constant')
    Y = np.pad(y, (0, N - len(y)), 'constant')
    
    X_fft = fft(X)
    Y_fft = fft(Y)
    
    Z_fft = [X_fft[i] * Y_fft[i] for i in range(N)]
    
    Z = ifft(Z_fft)
    return np.real(Z[:len(x) + len(y) - 1])

x = [1, 2, 3]
y = [4, 5, 6, 7, 8]

result = convolve(x, y)
print(result)
