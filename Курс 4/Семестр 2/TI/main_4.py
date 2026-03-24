import numpy as np
import random
import os
from itertools import product

def generate_matrix(n: int, m: int, seed: int = None) -> np.ndarray:
    """
    Генерирует порождающую матрицу G размера n x m вида [En | Dn,m-n].
    n — размерность кода (число строк), m — длина кодового слова (число столбцов).
    """
    if seed is not None:
        random.seed(seed)
    En = np.eye(n, dtype=int)
    D = np.array([[random.randint(0, 1) for _ in range(m - n)] for _ in range(n)], dtype=int)
    return np.hstack([En, D])


def save_matrix(G: np.ndarray, filepath: str):
    n, m = G.shape
    with open(filepath, "w") as f:
        f.write(f"{n} {m}\n")
        for row in G:
            f.write(" ".join(map(str, row)) + "\n")


def load_matrix(filepath: str) -> np.ndarray:
    with open(filepath) as f:
        n, m = map(int, f.readline().split())
        G = []
        for _ in range(n):
            row = list(map(int, f.readline().split()))
            assert len(row) == m, "Неверная длина строки матрицы"
            G.append(row)
    G = np.array(G, dtype=int)
    assert G.shape == (n, m), "Размер матрицы не совпадает с заголовком"
    return G


def all_codewords(G: np.ndarray) -> list:
    """
    Множество кодовых слов = {u * G (mod 2) | u ∈ GF(2)^n}.
    Количество слов = 2^n.
    """
    n, m = G.shape
    codewords = set()
    for u in product([0, 1], repeat=n):
        u_vec = np.array(u, dtype=int)
        codeword = tuple((u_vec @ G) % 2)
        codewords.add(codeword)
    return list(codewords)


def hamming_weight(v) -> int:
    return sum(v)


def min_distance(codewords: list) -> int:
    """
    Минимальное кодовое расстояние Хэмминга.
    Для линейного кода d_min = min{wt(c) | c ≠ 0}.
    """
    min_d = float("inf")
    zero = tuple([0] * len(codewords[0]))
    for c in codewords:
        if c != zero:
            w = hamming_weight(c)
            if w < min_d:
                min_d = w
    return int(min_d)



def code_parameters(G: np.ndarray) -> dict:
    n, m = G.shape
    codewords = all_codewords(G)
    num_codewords = len(codewords)
    d_min = min_distance(codewords)
    return {
        "n_rows": n,
        "code_length": m,
        "dimension": n,
        "num_codewords": num_codewords,
        "d_min": d_min,
    }



CONFIGS = [
    (2, 5),
    (3, 6),
    (4, 7),
    (3, 8),
    (5, 10),
]

os.makedirs("matrices", exist_ok=True)

print("=" * 65)
print(f"{'Код':<6} {'Длина n':>8} {'Разм. m':>9} {'Кол-во слов':>13} {'d_min':>7}")
print("=" * 65)

results = []
for idx, (n, m) in enumerate(CONFIGS, 1):
    filepath = f"matrices/code{idx}.txt"
    G = generate_matrix(n, m, seed=idx * 42)
    save_matrix(G, filepath)

    G_loaded = load_matrix(filepath)
    params = code_parameters(G_loaded)
    results.append(params)

    print(f"{'код' + str(idx):<6} {params['code_length']:>8} {params['dimension']:>9} "
          f"{params['num_codewords']:>13} {params['d_min']:>7}")

print("=" * 65)


print("\n--- Подробности по каждому коду ---\n")
for idx, (n, m) in enumerate(CONFIGS, 1):
    filepath = f"matrices/code{idx}.txt"
    G = load_matrix(filepath)
    print(f"Код {idx}: [{m},{n}]-код")
    print("Порождающая матрица G:")
    for row in G:
        print("  ", " ".join(map(str, row)))

    codewords = all_codewords(G)
    print(f"Кодовые слова ({len(codewords)} шт.):")
    for cw in sorted(codewords):
        print("  ", "".join(map(str, cw)))
    print()
