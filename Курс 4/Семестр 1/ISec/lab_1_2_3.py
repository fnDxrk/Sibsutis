import random


# ======= Быстрое возведение в степень по модулю ======= #
def mod_exp(a: int, x: int, p: int):
    y = 1
    s = a % p
    while x > 0:
        if x & 1:
            y = (y * s) % p
        s = (s * s) % p
        x >>= 1
    return y


# ======= Тест Ферма ======= #
def test_ferma(p: int, k: int = 5):
    if p == 2:
        return True
    if p % 2 == 0 or p < 2:
        return False

    for _ in range(k):
        a = random.randint(1, p - 1)
        if mod_exp(a, p - 1, p) != 1:
            return False
    return True


# ======= Расширенный алгоритм Евклида ======= #
def extended_euclid(a: int, b: int):
    U = [a, 1, 0]
    V = [b, 0, 1]

    while V[0] != 0:
        q = U[0] // V[0]
        T = [U[0] % V[0], U[1] - q * V[1], U[2] - q * V[2]]
        U, V = V, T

    return U[0], U[1], U[2]

# ======= Генерация простых чисел ======= #
def generate_prime(lower=100, upper=1000, k=5) -> int:
    while True:
        candidate = random.randint(lower, upper)
        if test_ferma(candidate, k):
            return candidate


# ======= Варианты получения (a,y,p) для дискр. логарифма ======= #
def get_ayp(mode: int):
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


# ======= Шаг младенца — шаг великана ======= #
def discrete_log_bsgs(a: int, y: int, p: int, max_solutions=10):
    if y == 1:
        return [0]
    if a % p == 0:
        return []

    m = int(p**0.5) + 1

    baby_steps = {}
    current = 1
    for j in range(m):
        baby_steps[current] = j
        current = (current * a) % p

    a_inv_m = mod_exp(a, p - 1 - m, p)

    order = p - 1

    solutions = []
    current = y

    for i in range(m):
        if current in baby_steps:
            j = baby_steps[current]
            x0 = i * m + j

            if mod_exp(a, x0, p) == y:
                k = 0
                while len(solutions) < max_solutions:
                    solution = x0 + k * order
                    solutions.append(solution)
                    k += 1
                break

        current = (current * a_inv_m) % p

    return solutions


# ======= Схема Диффи-Хеллмана ======= #
def diffie_hellman_key_exchange():
    print("\n=== Схема Диффи-Хеллмана ===")

    print("\nВыберите способ задания параметров:")
    print("1 - Ввести все параметры вручную")
    print("2 - Сгенерировать все параметры автоматически")
    print("3 - Ввести p и g, сгенерировать секретные ключи")

    mode = int(input("Ваш выбор: "))

    if mode == 1:
        p = int(input("Введите простое число p: "))
        g = int(input("Введите первообразный корень g: "))
        Xa = int(input("Введите секретный ключ абонента A (Xa): "))
        Xb = int(input("Введите секретный ключ абонента B (Xb): "))

    elif mode == 2:
        print("Генерация простого числа p...")
        p = generate_prime(10**8, 10**9)

        # Поиск первообразного корня g по модулю P
        print("Поиск корня g...")
        g = find_primitive_root(p)

        # Генерация секретных ключей
        Xa = random.randint(2, p - 2)
        Xb = random.randint(2, p - 2)

        print(f"Сгенерированные параметры:")
        print(f"p = {p}")
        print(f"g = {g}")
        print(f"Xa = {Xa}")
        print(f"Xb = {Xb}")

    elif mode == 3:
        # Полуручной режим
        p = int(input("Введите простое число p: "))
        g = int(input("Введите корень g: "))

        # Генерация секретных ключей
        Xa = random.randint(2, p - 2)
        Xb = random.randint(2, p - 2)

        print(f"Сгенерированные секретные ключи:")
        print(f"Xa = {Xa}")
        print(f"Xb = {Xb}")

    else:
        raise ValueError("Неверный режим")

    # Вычисление открытых ключей
    Ya = mod_exp(g, Xa, p)  # Ya = g^Xa mod p
    Yb = mod_exp(g, Xb, p)  # Yb = g^Xb mod p

    print(f"\nОткрытые ключи:")
    print(f"Ya = g^Xa mod p = {g}^{Xa} mod {p} = {Ya}")
    print(f"Yb = g^Xb mod p = {g}^{Xb} mod {p} = {Yb}")

    # Вычисление общего секретного ключа
    Kab_A = mod_exp(Yb, Xa, p)  # Kab = Yb^Xa mod p
    Kab_B = mod_exp(Ya, Xb, p)  # Kab = Ya^Xb mod p

    print(f"\nОбщий секретный ключ:")
    print(f"Kab (вычисленный A) = Yb^Xa mod p = {Yb}^{Xa} mod {p} = {Kab_A}")
    print(f"Kab (вычисленный B) = Ya^Xb mod p = {Ya}^{Xb} mod {p} = {Kab_B}")

    # Проверка совпадения ключей
    if Kab_A == Kab_B:
        print("✓ Ключи совпадают! Обмен успешен.")
        return Kab_A
    else:
        print("✗ Ошибка: ключи не совпадают!")
        return None


# ======= Находит первообразный корень по модулю p ======= #
def find_primitive_root(p):
    if p == 2:
        return 1

    # Факторизация p-1
    phi = p - 1
    factors = prime_factors(phi)

    for g in range(2, p):
        if all(mod_exp(g, phi // factor, p) != 1 for factor in factors):
            return g
    return None


# ======= Возвращает простые делители числа n ======= #
def prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


# ======= Основная программа ======= #
def main():
    print("Выберите метод:")
    print("1 - Возведение в степень по модулю")
    print("2 - Тест Ферма (проверка на простоту)")
    print("3 - Расширенный алгоритм Евклида")
    print("4 - Шаг младенца — шаг великана (дискретный логарифм)")
    print("5 - Схема Диффи-Хеллмана (общий ключ)")
    choice = int(input("Ваш выбор: "))

    if choice == 1:
        a = int(input("Введите a: "))
        x = int(input("Введите показатель степени x: "))
        p = int(input("Введите модуль p: "))
        print(f"{a}^{x} mod {p} = {mod_exp(a, x, p)}")
        return

    if choice == 5:
        diffie_hellman_key_exchange()
        return

    print("\nКак выбрать числа a и b?")
    print("1 - Ввести вручную")
    print("2 - Сгенерировать случайные")
    print("3 - Сгенерировать простые числа")
    num_choice = int(input("Ваш выбор: "))

    if num_choice == 1:
        a = int(input("Введите a: "))
        b = int(input("Введите b: "))
    elif num_choice == 2:
        a = random.randint(2, 1000)
        b = random.randint(2, 1000)
        print(f"Сгенерированы числа: a = {a}, b = {b}")
    else:
        a = generate_prime()
        b = generate_prime()
        print(f"Сгенерированы простые числа: a = {a}, b = {b}")

    if choice == 2:
        k = int(input("Введите количество повторов k: "))
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
        if x:
            print(f"x = {x}")
        else:
            print("Решение не найдено")


if __name__ == "__main__":
    main()
