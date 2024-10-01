import numpy as np
import time

# глобальные переменные для подсчета операций
operation_count_standard_multiplications = 0
operation_count_standard_additions = 0
operation_count_strassen_multiplications = 0
operation_count_strassen_additions = 0

# создание матриц A и B
def create_matrix_a(size):
    return np.array([[(-1) ** (i + j) for j in range(1, size + 1)] for i in range(1, size + 1)])

def create_matrix_b(size):
    return np.array([[i + j for j in range(1, size + 1)] for i in range(1, size + 1)])

# обычное умножение матриц
def standard_matrix_multiply(A, B):
    global operation_count_standard_multiplications, operation_count_standard_additions
    size = A.shape[0]
    C = np.zeros((size, size))  # нулевая матрица
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i, j] += A[i, k] * B[k, j]
                operation_count_standard_multiplications += 1  # считаем умножение
            operation_count_standard_additions += 1  # считаем сложение
    return C

# умножение матриц методом Штрассена
def strassen_matrix_multiply(A, B):
    global operation_count_strassen_multiplications, operation_count_strassen_additions
    n = A.shape[0]

    # если размер не является степенью двойки, дополняем до ближайшей степени двойки
    if n & (n - 1) != 0:  # если n не степень двойки
        new_size = 1 << (n - 1).bit_length()  # ближайшая степень двойки
        A = np.pad(A, ((0, new_size - n), (0, new_size - n)), mode='constant')
        B = np.pad(B, ((0, new_size - n), (0, new_size - n)), mode='constant')

    n = A.shape[0]  # обновляем n после дополнения

    if n <= 2:
        return standard_matrix_multiply(A, B)

    # делим на 4 подматрицы
    mid = n // 2
    A1, A2 = A[:mid, :mid], A[:mid, mid:]  # A1 A2
    A3, A4 = A[mid:, :mid], A[mid:, mid:]  # A3 A4
    B1, B2 = B[:mid, :mid], B[:mid, mid:]  # B1 B2
    B3, B4 = B[mid:, :mid], B[mid:, mid:]  # B3 B4

    # шаги Штрассена
    M1 = strassen_matrix_multiply(A2 - A4, B3 + B4)
    operation_count_strassen_additions += mid * mid * 2  
    operation_count_strassen_multiplications += mid * mid  
    M2 = strassen_matrix_multiply(A1 + A4, B1 + B4)
    operation_count_strassen_additions += mid * mid * 2
    operation_count_strassen_multiplications += mid * mid
    M3 = strassen_matrix_multiply(A1 - A3, B1 + B2)
    operation_count_strassen_additions += mid * mid * 2
    operation_count_strassen_multiplications += mid * mid
    M4 = strassen_matrix_multiply(A1 + A2, B4)
    operation_count_strassen_additions += mid * mid
    operation_count_strassen_multiplications += mid * mid
    M5 = strassen_matrix_multiply(A1, B2 - B4)
    operation_count_strassen_additions += mid * mid * 1
    operation_count_strassen_multiplications += mid * mid
    M6 = strassen_matrix_multiply(A4, B3 - B1)
    operation_count_strassen_additions += mid * mid * 1
    operation_count_strassen_multiplications += mid * mid
    M7 = strassen_matrix_multiply(A3 + A4, B1)
    operation_count_strassen_additions += mid * mid * 1
    operation_count_strassen_multiplications += mid * mid

    C1 = M1 + M2 - M4 + M6
    operation_count_strassen_additions += mid * mid * 3
    C2 = M4 + M5
    operation_count_strassen_additions += mid * mid
    C3 = M6 + M7
    operation_count_strassen_additions += mid * mid
    C4 = M2 - M3 + M5 - M7
    operation_count_strassen_additions += mid * mid * 3

    # получаем итоговую матрицу
    C = np.zeros((n, n))
    C[:mid, :mid] = C1
    C[:mid, mid:] = C2
    C[mid:, :mid] = C3
    C[mid:, mid:] = C4

    return C[:n, :n]  # возвращаем результат в исходном размере

# размер матриц
N = 128

# создание матриц
A = create_matrix_a(N)
B = create_matrix_b(N)

# измерение времени для обычного умножения
start_time = time.time()
C_standard = standard_matrix_multiply(A, B)
end_time = time.time()
print("Стандартное умножение завершено за:", end_time - start_time, "секунд")
print("Количество операций (стандартное умножение):")
print("Умножения:", operation_count_standard_multiplications)
print("Сложения:", operation_count_standard_additions)

# сброс счётчика операций
operation_count_strassen_multiplications = 0
operation_count_strassen_additions = 0

# измерение времени для метода Штрассена
start_time = time.time()
C_strassen = strassen_matrix_multiply(A, B)
end_time = time.time()
print("Умножение методом Штрассена завершено за:", end_time - start_time, "секунд")
print("Количество операций (метод Штрассена):")
print("Умножения:", operation_count_strassen_multiplications)
print("Сложения:", operation_count_strassen_additions)

if np.array_equal(C_standard, C_strassen[:C_standard.shape[0], :C_standard.shape[1]]):
    print("Результаты совпадают!")
else:
    print("Результаты не совпадают!")
