import random


def mod_exp(a: int, x: int, p: int):
    """Быстрое возведение в степень по модулю"""
    y = 1
    s = a % p
    while x > 0:
        if x & 1:
            y = (y * s) % p
        s = (s * s) % p
        x >>= 1
    return y


def test_ferma(p: int, k: int = 5):
    """Тест Ферма на простоту"""
    if p == 2:
        return True
    if p % 2 == 0 or p < 2:
        return False

    for _ in range(k):
        a = random.randint(1, p - 1)
        if mod_exp(a, p - 1, p) != 1:
            return False
    return True


def extended_euclid(a: int, b: int):
    """Обобщенный алгоритм Евклида: возвращает gcd, x, y"""
    U = [a, 1, 0]
    V = [b, 0, 1]
    while V[0] != 0:
        q = U[0] // V[0]
        T = [U[0] % V[0], U[1] - q * V[1], U[2] - q * V[2]]
        U, V = V, T
    return U[0], U[1], U[2]


def generate_prime(lower=100, upper=1000, k=5):
    """Генерация случайного простого числа в заданном диапазоне"""
    while True:
        candidate = random.randint(lower, upper)
        if test_ferma(candidate, k):
            return candidate


def get_valid_input(prompt: str, valid_range=None):
    """Получение валидного целочисленного ввода от пользователя"""
    while True:
        try:
            value = int(input(prompt))
            if valid_range and value not in valid_range:
                print(f"Ошибка: Введите число из {valid_range}!")
                continue
            return value
        except ValueError:
            print("Ошибка: Введите целое число!")


def get_numbers(num_choice: int):
    """Получение или генерация чисел a и b"""
    if num_choice == 1:
        a = get_valid_input("Введите a: ")
        b = get_valid_input("Введите b: ")
    elif num_choice == 2:
        a = random.randint(2, 1000)
        b = random.randint(2, 1000)
        print(f"Сгенерированы числа: a = {a}, b = {b}")
    else:
        a = generate_prime()
        b = generate_prime()
        print(f"Сгенерированы простые числа: a = {a}, b = {b}")
    return a, b


def get_ayp(mode: int):
    """
    Варианты получения (a,y,p) для дискр. логарифма
    mode = 1 -> ввод с клавиатуры
    mode = 2 -> генерация случайных чисел
    mode = 3 -> генерация случайных простых p
    """
    if mode == 1:
        a = int(input("Введите a: "))
        y = int(input("Введите y: "))
        p = int(input("Введите p: "))
    elif mode == 2:
        a = random.randint(2, 10)
        y = random.randint(2, 50)
        p = random.randint(50, 200)
    elif mode == 3:

        def gen_prime():
            while True:
                n = random.randint(50, 200)
                if test_ferma(n, 7):
                    return n

        p = gen_prime()
        a = random.randint(2, p - 2)
        y = random.randint(2, p - 2)
    else:
        raise ValueError("Неверный режим")
    return a, y, p


def discrete_log_bsgs(a: int, y: int, p: int):
    """
    Решает уравнение a^x ≡ y (mod p)
    методом Шаг младенца — шаг великан
    с трудоёмкостью O(p log2 p)
    """
    m = p  # почти полный диапазон (даёт O(p log2 p))
    baby = []  # (значение, j)
    cur = y % p
    for j in range(m):
        baby.append((cur, j))
        cur = (cur * a) % p

    baby.sort(key=lambda x: x[0])  # сортировка O(p log2 p)

    a_m = mod_exp(a, m, p)
    cur = 1
    for i in range(1, 3):  # i достаточно 1-2, так как m ≈ p
        cur = (cur * a_m) % p
        # бинарный поиск по отсортированному baby
        lo, hi = 0, len(baby) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if baby[mid][0] < cur:
                lo = mid + 1
            elif baby[mid][0] > cur:
                hi = mid - 1
            else:
                # найдено совпадение
                j = baby[mid][1]
                x = i * m - j
                return x % (p - 1)
    return None


def print_menu():
    """Вывод меню выбора метода"""
    print("\nВыберите метод:")
    print("1 - Возведение в степень по модулю")
    print("2 - Тест простоты Ферма")
    print("3 - Расширенный алгоритм Евклида")


def print_number_selection_menu():
    """Вывод меню выбора способа получения чисел"""
    print("\nКак выбрать числа a и b?")
    print("1 - Ввести вручную")
    print("2 - Сгенерировать случайные")
    print("3 - Сгенерировать простые числа")


def main():
    print_menu()
    choice = get_valid_input("Ваш выбор: ", {1, 2, 3})

    if choice == 1:
        a = get_valid_input("Введите a: ")
        x = get_valid_input("Введите показатель степени x: ")
        p = get_valid_input("Введите модуль p: ")
        print(f"{a}^{x} mod {p} = {mod_exp(a, x, p)}")
        return

    print_number_selection_menu()
    num_choice = get_valid_input("Ваш выбор: ", {1, 2, 3})
    a, b = get_numbers(num_choice)

    if choice == 2:
        k = get_valid_input("Введите количество повторов k: ")
        print(f"{a} {'— простое' if test_ferma(a, k) else '— составное'}")
        print(f"{b} {'— простое' if test_ferma(b, k) else '— составное'}")
    elif choice == 3:
        g, x, y = extended_euclid(a, b)
        print(f"НОД({a}, {b}) = {g}")
        print(f"{a}*({x}) + {b}*({y}) = {a * x + b * y}")

    elif choice == 4:
        p = int(input("Введите модуль p: "))
        print(f"Решается уравнение: {a}^x ≡ {b} (mod {p})")
        x = discrete_log_bsgs(a, b, p)
        if x is not None:
            print(f"x = {x}")
        else:
            print("Решение не найдено")


if __name__ == "__main__":
    main()
