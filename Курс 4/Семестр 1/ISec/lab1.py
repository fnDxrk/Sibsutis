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


if __name__ == "__main__":
    main()
