import math
from typing import List

class Fraction:
    def __init__(self, numerator: int = 0, denominator: int = 1):
        if denominator == 0:
            raise ValueError("Знаменатель не может быть равен 0")

        gcd_val = math.gcd(abs(numerator), abs(denominator))
        self.numerator = numerator // gcd_val
        self.denominator = denominator // gcd_val

        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator

    def __str__(self):
        return f"{self.numerator}" if self.denominator == 1 else f"{self.numerator}/{self.denominator}"

    def __add__(self, other):
        return Fraction(self.numerator * other.denominator + other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __sub__(self, other):
        return Fraction(self.numerator * other.denominator - other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __mul__(self, other):
        return Fraction(self.numerator * other.numerator, self.denominator * other.denominator)

    def __truediv__(self, other):
        return Fraction(self.numerator * other.denominator, self.denominator * other.numerator)


def read_matrix(filename: str) -> List[List[Fraction]]:
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = [Fraction(int(float(value) * 1000), 1000) for value in line.split()]
            matrix.append(row)
    return matrix


def print_matrix(matrix: List[List[Fraction]]):
    for row in matrix:
        print(" ".join(f"{str(value):>10}" for value in row))
    print()


def gauss_jordan(matrix: List[List[Fraction]]):
    m = len(matrix)
    if m == 0:
        print("Матрица пуста")
        return
    n = len(matrix[0]) - 1
    k = 0
    leading_cols = []
    step = 1

    for col in range(n):
        if k >= m:
            break

        max_row = k
        max_val = matrix[k][col]
        for i in range(k + 1, m):
            if abs(matrix[i][col].numerator) > abs(max_val.numerator):
                max_val = matrix[i][col]
                max_row = i

        if max_val.numerator == 0:
            continue

        if max_row != k:
            matrix[k], matrix[max_row] = matrix[max_row], matrix[k]

        divisor = matrix[k][col]
        for j in range(col, n + 1):
            matrix[k][j] = matrix[k][j] / divisor

        for i in range(m):
            if i != k:
                factor = matrix[i][col]
                for j in range(col, n + 1):
                    matrix[i][j] = matrix[i][j] - factor * matrix[k][j]

        print(f"После шага {step}:")
        print_matrix(matrix)
        step += 1
        leading_cols.append(col)
        k += 1

    has_inconsistent = any(
        all(matrix[i][j].numerator == 0 for j in range(n)) and matrix[i][n].numerator != 0 for i in range(m))
    if has_inconsistent:
        print("Система не имеет решений")
        return

    num_leading = len(leading_cols)
    if num_leading == n:
        solution = [Fraction() for _ in range(n)]
        for i in range(num_leading):
            solution[leading_cols[i]] = matrix[i][n]
        print("Единственное решение:")
        for i, sol in enumerate(solution):
            print(f"x{i} = {sol}")
    else:
        free_vars = [j for j in range(n) if j not in leading_cols]
        print("Система имеет бесконечно много решений")
        print("Свободные переменные:", " ".join(f"x{j}" for j in free_vars))

        for i in range(num_leading):
            lead_col = leading_cols[i]
            expr = f"x{lead_col} = {matrix[i][n]}"
            for fv in free_vars:
                coeff = matrix[i][fv]
                if coeff.numerator != 0:
                    sign = "-" if coeff.numerator > 0 else "+"
                    expr += f" {sign} {abs(coeff.numerator)}/{coeff.denominator}*x{fv}"
            print(expr)


if __name__ == "__main__":
    filename = "matrix.txt"
    matrix = read_matrix(filename)
    print("Исходная матрица:")
    print_matrix(matrix)
    gauss_jordan(matrix)
