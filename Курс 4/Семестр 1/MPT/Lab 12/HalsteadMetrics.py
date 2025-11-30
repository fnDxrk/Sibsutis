import tokenize
import io
import keyword
import builtins
import math
from typing import List, Tuple, Dict, Any
import inspect

S = 18


class HalsteadMetrics:
    def __init__(self):
        self.python_keywords = set(keyword.kwlist)
        self.builtin_names = set(dir(builtins))

    def extract_tokens(self, code: str) -> Tuple[List[str], List[str]]:
        """Извлекает операторы и операнды из кода функции"""
        operators = []
        operands = []

        try:
            f = io.BytesIO(code.encode("utf-8"))
            for tok in tokenize.tokenize(f.readline):
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                ):
                    continue

                token_str = tok.string

                if tok.type == tokenize.OP:
                    operators.append(token_str)
                elif tok.type == tokenize.NAME:
                    if token_str in self.python_keywords:
                        operators.append(token_str)
                    else:
                        operands.append(token_str)
                elif tok.type in (tokenize.NUMBER, tokenize.STRING):
                    operands.append(token_str)

        except tokenize.TokenError:
            pass

        return operators, operands

    def compute_metrics(self, code: str, eta2_star: int) -> Dict[str, Any]:
        """Вычисляет все метрики Холстеда для заданного кода"""
        operators, operands = self.extract_tokens(code)

        unique_operators = set(operators)
        unique_operands = set(operands)

        eta1 = len(unique_operators)
        eta2 = len(unique_operands)
        eta = eta1 + eta2

        N1 = len(operators)
        N2 = len(operands)
        N = N1 + N2

        N_hat = 0.0
        if eta1 > 0:
            N_hat += eta1 * math.log2(eta1)
        if eta2 > 0:
            N_hat += eta2 * math.log2(eta2)

        V_star = (2 + eta2_star) * math.log2(2 + eta2_star) if eta2_star > 0 else 0

        V = N * math.log2(eta) if eta > 0 else 0

        L = V_star / V if V > 0 else 0
        L_hat = (2 / eta1) * (eta2 / N2) if (eta1 > 0 and N2 > 0) else 0

        I = 0.0
        if eta1 > 0 and N1 > 0 and (eta1 + eta2) > 0:
            I = (2 / eta1) * (N2 / N1) * (eta1 + eta2) * math.log2(eta1 + eta2)

        T1 = V_star / S if S > 0 else 0

        T2 = 0.0
        if S > 0 and eta1 > 0 and eta2 > 0:
            T2 = (N_hat * (eta1 * math.log2(eta2) + eta2 * math.log2(eta1))) / (2 * S)

        T3 = 0.0
        if S > 0 and N1 > 0 and N2 > 0 and eta2 > 0:
            T3 = (N1 * N2 * math.log2(eta2)) / (2 * S)

        return {
            "eta2_star": eta2_star,
            "eta1": eta1,
            "eta2": eta2,
            "eta": eta,
            "N1": N1,
            "N2": N2,
            "N": N,
            "N_hat": N_hat,
            "V_star": V_star,
            "V": V,
            "L": L,
            "L_hat": L_hat,
            "I": I,
            "T1": T1,
            "T2": T2,
            "T3": T3,
        }


def task1_find_min(arr: List[int]) -> Tuple[int, int]:
    """1. Минимальный элемент одномерного массива и его индекс"""
    if not arr:
        raise ValueError("Array is empty")
    min_val = arr[0]
    min_idx = 0
    for i in range(1, len(arr)):
        if arr[i] < min_val:
            min_val = arr[i]
            min_idx = i
    return min_val, min_idx


def task2_bubble_sort(arr: List[int]) -> List[int]:
    """2. Сортировка пузырьком"""
    n = len(arr)
    arr = arr[:]
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def task3_binary_search(arr: List[int], target: int) -> int:
    """3. Бинарный поиск"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def task4_find_min_2d(matrix: List[List[int]]) -> Tuple[int, int, int]:
    """4. Минимальный элемент двумерного массива"""
    if not matrix or not matrix[0]:
        raise ValueError("Matrix is empty")
    min_val = matrix[0][0]
    min_i = min_j = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < min_val:
                min_val = matrix[i][j]
                min_i, min_j = i, j
    return min_val, min_i, min_j


def task5_reverse_array(arr: List[int]) -> List[int]:
    """5. Перестановка в обратном порядке"""
    return arr[::-1]


def task6_cyclic_shift_left(arr: List[int], k: int) -> List[int]:
    """6. Циклический сдвиг влево на k позиций"""
    if not arr:
        return arr
    k = k % len(arr)
    return arr[k:] + arr[:k]


def task7_replace_value(arr: List[int], old_val: int, new_val: int) -> List[int]:
    """7. Замена всех вхождений значения"""
    return [new_val if x == old_val else x for x in arr]


TASKS = [
    (task1_find_min, 3),
    (task2_bubble_sort, 2),
    (task3_binary_search, 3),
    (task4_find_min_2d, 4),
    (task5_reverse_array, 2),
    (task6_cyclic_shift_left, 3),
    (task7_replace_value, 4),
]


def get_function_source(func) -> str:
    """Получает исходный код тела функции (без def и docstring)"""
    try:
        lines = inspect.getsourcelines(func)[0]
        code_lines = []
        after_def = False
        skip_docstring = False

        for line in lines:
            stripped = line.strip()

            if not after_def:
                if stripped.endswith(":"):
                    after_def = True
                continue

            if stripped.startswith('"""') or stripped.startswith("'''"):
                skip_docstring = not skip_docstring
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    skip_docstring = False
                continue

            if skip_docstring:
                continue

            if stripped == "":
                continue

            code_lines.append(line)

        return "".join(code_lines)
    except:
        return inspect.getsource(func)


def main():
    metrics_calculator = HalsteadMetrics()

    all_L_hat = []
    all_V = []

    print("=" * 100)
    print("ЛАБОРАТОРНАЯ РАБОТА №12: МЕТРИКИ ХОЛСТЕДА")
    print("=" * 100)

    for i, (func, eta2_star) in enumerate(TASKS, 1):
        print(
            f"\n--- ЗАДАЧА {i}: {func.__doc__.strip() if func.__doc__ else 'Без описания'} ---"
        )

        code = get_function_source(func)
        metrics = metrics_calculator.compute_metrics(code, eta2_star)

        print(f"η₂* (смысловые параметры): {metrics['eta2_star']}")
        print(f"η₁ (уникальные операторы): {metrics['eta1']}")
        print(f"η₂ (уникальные операнды): {metrics['eta2']}")
        print(f"η (словарь): {metrics['eta']}")
        print(f"N₁ (вхождения операторов): {metrics['N1']}")
        print(f"N₂ (вхождения операндов): {metrics['N2']}")
        print(f"N (длина реализации): {metrics['N']}")
        print(f"Ń (предсказанная длина): {metrics['N_hat']:.2f}")
        print(f"V* (потенциальный объём): {metrics['V_star']:.2f}")
        print(f"V (объём реализации): {metrics['V']:.2f}")
        print(f"L (уровень через V*): {metrics['L']:.4f}")
        print(f"L̂ (уровень по реализации): {metrics['L_hat']:.4f}")
        print(f"I (интеллектуальное содержание): {metrics['I']:.2f}")
        print(f"T̂₁ (время по V*): {metrics['T1']:.2f} сек")
        print(f"T̂₂ (время по Ń): {metrics['T2']:.2f} сек")
        print(f"T̂₃ (время по реализации): {metrics['T3']:.2f} сек")

        all_L_hat.append(metrics["L_hat"])
        all_V.append(metrics["V"])

    n = len(all_L_hat)
    if n > 0:
        lambda1 = sum(l_hat * v for l_hat, v in zip(all_L_hat, all_V)) / (2 * n)
        lambda2 = sum(v * v for v in all_V) / (2 * n)

        print("\n" + "=" * 100)
        print("СРЕДНИЕ ЗНАЧЕНИЯ УРОВНЕЙ ЯЗЫКА ПРОГРАММИРОВАНИЯ:")
        print(f"λ₁ = (Σ L̂ᵢ·Vᵢ) / (2·n) = {lambda1:.2f}")
        print(f"λ₂ = (Σ Vᵢ²) / (2·n) = {lambda2:.2f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
