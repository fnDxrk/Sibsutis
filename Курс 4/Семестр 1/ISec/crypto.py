import random
import math
import os
from typing import Tuple, Optional, Set


class CryptoUtils:
    """Утилиты для криптографических операций"""

    @staticmethod
    def mod_exp(a: int, x: int, p: int) -> int:
        """Быстрое возведение в степень по модулю"""
        y = 1
        s = a % p
        while x > 0:
            if x & 1:
                y = (y * s) % p
            s = (s * s) % p
            x >>= 1
        return y

    @staticmethod
    def test_ferma(p: int, k: int = 5) -> bool:
        """Тест Ферма на простоту"""
        if p == 2:
            return True
        if p % 2 == 0 or p < 2:
            return False

        for _ in range(k):
            a = random.randint(1, p - 1)
            if CryptoUtils.mod_exp(a, p - 1, p) != 1:
                return False
        return True

    @staticmethod
    def extended_euclid(a: int, b: int) -> Tuple[int, int, int]:
        """Расширенный алгоритм Евклида"""
        U = [a, 1, 0]
        V = [b, 0, 1]

        while V[0] != 0:
            q = U[0] // V[0]
            T = [U[0] % V[0], U[1] - q * V[1], U[2] - q * V[2]]
            U, V = V, T

        return U[0], U[1], U[2]

    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Нахождение обратного элемента по модулю"""
        g, x, _ = CryptoUtils.extended_euclid(a, m)
        if g != 1:
            raise ValueError(f"Обратный элемент не существует для a={a}, m={m}")
        return x % m

    @staticmethod
    def generate_prime(lower: int = 100, upper: int = 1000, k: int = 5) -> int:
        """Генерация простого числа в заданном диапазоне"""
        while True:
            candidate = random.randint(lower, upper)
            if CryptoUtils.test_ferma(candidate, k):
                return candidate

    @staticmethod
    def generate_large_prime(lower: int = 10 ** 8, upper: int = 10 ** 9, k: int = 10) -> int:
        """Генерация большого простого числа"""
        while True:
            candidate = random.randint(lower, upper)
            if CryptoUtils.test_ferma(candidate, k):
                return candidate

    @staticmethod
    def prime_factors(n: int) -> Set[int]:
        """Разложение числа на простые множители"""
        factors: Set[int] = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    @staticmethod
    def find_primitive_root(p: int) -> int:
        """Поиск примитивного корня по модулю p"""
        if p == 2:
            return 1

        phi = p - 1
        factors = CryptoUtils.prime_factors(phi)

        for g in range(2, min(p, 10000)):
            if all(CryptoUtils.mod_exp(g, phi // factor, p) != 1 for factor in factors):
                return g
        return 2

    @staticmethod
    def generate_coprime(phi: int) -> int:
        """Генерация числа, взаимно простого с phi"""
        while True:
            candidate = random.randint(2, phi - 1)
            if math.gcd(candidate, phi) == 1:
                return candidate

    @staticmethod
    def calculate_block_size(n: int) -> int:
        """Вычисление размера блока для шифрования"""
        block_size_bits = n.bit_length() - 1
        block_size_bytes = max(1, block_size_bits // 8)
        return block_size_bytes


class RSACrypto:
    """Класс для работы с RSA шифрованием"""

    def __init__(self):
        self.utils = CryptoUtils()

    def generate_keys(self, mode: int = 1) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """Генерация ключей для RSA"""
        print("\n=== Генерация ключей RSA ===")

        if mode == 1:
            p = int(input("Введите простое число p: "))
            q = int(input("Введите простое число q: "))
            d = int(input("Введите секретный ключ d: "))

            if not self.utils.test_ferma(p):
                raise ValueError(f"Число p={p} не является простым!")
            if not self.utils.test_ferma(q):
                raise ValueError(f"Число q={q} не является простым!")

        elif mode == 2:
            print("Генерация простых чисел p и q...")
            p = self.utils.generate_large_prime()
            q = self.utils.generate_large_prime()
            while q == p:
                q = self.utils.generate_large_prime()

            print(f"Сгенерированные простые числа:")
            print(f"p = {p}")
            print(f"q = {q}")

            phi = (p - 1) * (q - 1)
            d = self.utils.generate_coprime(phi)
            print(f"d = {d}")

        else:
            raise ValueError("Неверный режим")

        n = p * q
        phi = (p - 1) * (q - 1)
        c = self.utils.mod_inverse(d, phi)

        print(f"\nВычисленные параметры:")
        print(f"n = p * q = {p} * {q} = {n}")
        print(f"φ(n) = (p-1)*(q-1) = {phi}")
        print(f"c = d^(-1) mod φ(n) = {c}")

        public_key = (n, c)
        private_key = (n, d, p, q)

        return public_key, private_key

    def encrypt_file(self, public_key: Tuple[int, int], input_file: str, output_file: str):
        """Шифрование файла с помощью RSA"""
        n, c = public_key
        block_size = self.utils.calculate_block_size(n)

        print(f"Размер блока: {block_size} байт")

        with open(input_file, 'rb') as f:
            data = f.read()

        encrypted_blocks = []

        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]

            if len(block) < block_size:
                block = block.ljust(block_size, b'\x00')

            m = int.from_bytes(block, 'big')

            if m >= n:
                raise ValueError(f"Блок данных слишком большой для модуля n={n}")

            e = self.utils.mod_exp(m, c, n)
            encrypted_blocks.append(e)

        with open(output_file, 'wb') as f:
            f.write(block_size.to_bytes(4, 'big'))
            f.write(len(encrypted_blocks).to_bytes(8, 'big'))

            for block in encrypted_blocks:
                block_bytes = block.to_bytes((n.bit_length() + 7) // 8, 'big')
                f.write(len(block_bytes).to_bytes(2, 'big'))
                f.write(block_bytes)

        print(f"Файл зашифрован. Размер исходного файла: {len(data)} байт")
        print(f"Количество блоков: {len(encrypted_blocks)}")

    def decrypt_file(self, private_key: Tuple[int, int, int, int], input_file: str, output_file: str):
        """Дешифрование файла с помощью RSA"""
        n, d, p, q = private_key

        with open(input_file, 'rb') as f:
            block_size = int.from_bytes(f.read(4), 'big')
            num_blocks = int.from_bytes(f.read(8), 'big')

            encrypted_blocks = []
            for _ in range(num_blocks):
                block_len = int.from_bytes(f.read(2), 'big')
                block = int.from_bytes(f.read(block_len), 'big')
                encrypted_blocks.append(block)

        decrypted_data = bytearray()

        for encrypted_block in encrypted_blocks:
            m = self.utils.mod_exp(encrypted_block, d, n)
            decrypted_block = m.to_bytes(block_size, 'big')
            decrypted_data.extend(decrypted_block)

        decrypted_data = decrypted_data.rstrip(b'\x00')

        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

        print(f"Файл расшифрован. Размер: {len(decrypted_data)} байт")


class DiffieHellman:
    """Класс для схемы Диффи-Хеллмана"""

    def __init__(self):
        self.utils = CryptoUtils()

    def key_exchange(self) -> Optional[int]:
        """Схема Диффи-Хеллмана для обмена ключами"""
        print("\n=== Схема Диффи-Хеллмана ===")

        print("\nВыберите способ задания параметров:")
        print("1 - Ввести все параметры вручную")
        print("2 - Сгенерировать все параметры автоматически")
        print("3 - Ввести p и g, сгенерировать секретные ключи")

        mode = int(input("Ваш выбор: "))

        if mode == 1:
            p = int(input("Введите простое число p: "))
            g = int(input("Введите генератор g: "))
            Xa = int(input("Введите секретный ключ абонента A (Xa): "))
            Xb = int(input("Введите секретный ключ абонента B (Xb): "))

        elif mode == 2:
            print("Генерация простого числа p...")
            p = self.utils.generate_large_prime()
            print("Поиск генератора g...")
            g = self.utils.find_primitive_root(p)
            Xa = random.randint(2, p - 2)
            Xb = random.randint(2, p - 2)

            print(f"Сгенерированные параметры:")
            print(f"p = {p}")
            print(f"g = {g}")
            print(f"Xa = {Xa}")
            print(f"Xb = {Xb}")

        elif mode == 3:
            p = int(input("Введите простое число p: "))
            g = int(input("Введите генератор g: "))
            Xa = random.randint(2, p - 2)
            Xb = random.randint(2, p - 2)

            print(f"Сгенерированные секретные ключи:")
            print(f"Xa = {Xa}")
            print(f"Xb = {Xb}")

        else:
            print("Неверный режим")
            return None

        Ya = self.utils.mod_exp(g, Xa, p)
        Yb = self.utils.mod_exp(g, Xb, p)

        print(f"\nОткрытые ключи:")
        print(f"Ya = g^Xa mod p = {g}^{Xa} mod {p} = {Ya}")
        print(f"Yb = g^Xb mod p = {g}^{Xb} mod {p} = {Yb}")

        Kab_A = self.utils.mod_exp(Yb, Xa, p)
        Kab_B = self.utils.mod_exp(Ya, Xb, p)

        print(f"\nОбщий секретный ключ:")
        print(f"Kab (вычисленный A) = Yb^Xa mod p = {Yb}^{Xa} mod {p} = {Kab_A}")
        print(f"Kab (вычисленный B) = Ya^Xb mod p = {Ya}^{Xb} mod {p} = {Kab_B}")

        if Kab_A == Kab_B:
            print("✓ Ключи совпадают! Обмен успешен.")
            return Kab_A
        else:
            print("✗ Ошибка: ключи не совпадают!")
            return None


class ElGamalCrypto:
    """Класс для шифра Эль-Гамаля"""

    def __init__(self):
        self.utils = CryptoUtils()

    def generate_keys(self, mode: int = 1) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
        """Генерация ключей для шифра Эль-Гамаля"""
        print("\n=== Генерация ключей Эль-Гамаля ===")

        if mode == 1:
            p = int(input("Введите простое число p: "))
            g = int(input("Введите генератор g: "))
            C1 = int(input("Введите открытый ключ C1: "))
            D1 = random.randint(2, p - 2)

        elif mode == 2:
            print("Генерация простого числа p...")
            p = self.utils.generate_large_prime()
            print("Поиск генератора g...")
            g = self.utils.find_primitive_root(p)
            D1 = random.randint(2, p - 2)
            C1 = self.utils.mod_exp(g, D1, p)

            print(f"Сгенерированные параметры:")
            print(f"p = {p}")
            print(f"g = {g}")
            print(f"C1 = {C1}")
            print(f"D1 = {D1}")

        else:
            raise ValueError("Неверный режим")

        public_key = (p, g, C1)
        private_key = (p, D1)
        return public_key, private_key

    def encrypt_file(self, public_key: Tuple[int, int, int], input_file: str, output_file: str):
        """Шифрование файла с помощью Эль-Гамаля"""
        p, g, C1 = public_key
        block_size = self.utils.calculate_block_size(p)

        print(f"Размер блока: {block_size} байт")

        with open(input_file, 'rb') as f:
            data = f.read()

        encrypted_blocks = []

        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]

            if len(block) < block_size:
                block = block.ljust(block_size, b'\x00')

            m = int.from_bytes(block, 'big')
            if m >= p:
                m = m % p

            k = random.randint(2, p - 2)
            a = self.utils.mod_exp(g, k, p)
            b = (m * self.utils.mod_exp(C1, k, p)) % p

            encrypted_blocks.append((a, b))

        with open(output_file, 'wb') as f:
            f.write(block_size.to_bytes(4, 'big'))
            f.write(len(encrypted_blocks).to_bytes(8, 'big'))

            for a, b in encrypted_blocks:
                a_bytes = a.to_bytes((p.bit_length() + 7) // 8, 'big')
                b_bytes = b.to_bytes((p.bit_length() + 7) // 8, 'big')

                f.write(len(a_bytes).to_bytes(2, 'big'))
                f.write(a_bytes)
                f.write(len(b_bytes).to_bytes(2, 'big'))
                f.write(b_bytes)

        print(f"Файл зашифрован. Размер исходного файла: {len(data)} байт")
        print(f"Количество блоков: {len(encrypted_blocks)}")

    def decrypt_file(self, private_key: Tuple[int, int], input_file: str, output_file: str):
        """Дешифрование файла с помощью Эль-Гамаля"""
        p, D1 = private_key

        with open(input_file, 'rb') as f:
            block_size = int.from_bytes(f.read(4), 'big')
            num_blocks = int.from_bytes(f.read(8), 'big')

            encrypted_blocks = []
            for _ in range(num_blocks):
                a_len = int.from_bytes(f.read(2), 'big')
                a = int.from_bytes(f.read(a_len), 'big')
                b_len = int.from_bytes(f.read(2), 'big')
                b = int.from_bytes(f.read(b_len), 'big')
                encrypted_blocks.append((a, b))

        decrypted_data = bytearray()

        for a, b in encrypted_blocks:
            s = self.utils.mod_exp(a, D1, p)
            s_inv = self.utils.mod_inverse(s, p)
            m = (b * s_inv) % p

            decrypted_block = m.to_bytes(block_size, 'big')
            decrypted_data.extend(decrypted_block)

        decrypted_data = decrypted_data.rstrip(b'\x00')

        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

        print(f"Файл расшифрован. Размер: {len(decrypted_data)} байт")


class ShamirCrypto:
    """Класс для шифра Шамира"""

    def __init__(self):
        self.utils = CryptoUtils()

    def generate_keys(self, mode: int = 1) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Генерация ключей для шифра Шамира"""
        print("\n=== Генерация ключей Шамира ===")

        if mode == 1:
            p = int(input("Введите простое число p: "))
            Ca = int(input("Введите открытый ключ абонента A (Ca): "))
            Cb = int(input("Введите открытый ключ абонента B (Cb): "))

            if math.gcd(Ca, p - 1) != 1:
                raise ValueError("Ca должен быть взаимно прост с p-1")
            if math.gcd(Cb, p - 1) != 1:
                raise ValueError("Cb должен быть взаимно прост с p-1")

            Da = self.utils.mod_inverse(Ca, p - 1)
            Db = self.utils.mod_inverse(Cb, p - 1)

        elif mode == 2:
            p = self.utils.generate_large_prime()

            while True:
                Ca = random.randint(2, p - 2)
                if math.gcd(Ca, p - 1) == 1:
                    break

            while True:
                Cb = random.randint(2, p - 2)
                if math.gcd(Cb, p - 1) == 1:
                    break

            Da = self.utils.mod_inverse(Ca, p - 1)
            Db = self.utils.mod_inverse(Cb, p - 1)

            print(f"Сгенерированные параметры:")
            print(f"p = {p}")
            print(f"Ca = {Ca}")
            print(f"Cb = {Cb}")
            print(f"Da = {Da}")
            print(f"Db = {Db}")

        else:
            raise ValueError("Неверный режим")

        keys_a = (p, Ca, Da)
        keys_b = (p, Cb, Db)
        return keys_a, keys_b

    def encrypt_file(self, keys_a: Tuple[int, int, int], keys_b: Tuple[int, int, int],
                     input_file: str, output_file: str):
        """Шифрование файла с помощью шифра Шамира"""
        p, Ca, Da = keys_a
        p, Cb, Db = keys_b
        block_size = self.utils.calculate_block_size(p)

        print(f"Размер блока: {block_size} байт")

        with open(input_file, 'rb') as f:
            data = f.read()

        encrypted_blocks = []

        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]

            if len(block) < block_size:
                block = block.ljust(block_size, b'\x00')

            m = int.from_bytes(block, 'big')
            if m >= p:
                m = m % p

            x1 = self.utils.mod_exp(m, Ca, p)
            x2 = self.utils.mod_exp(x1, Cb, p)
            x3 = self.utils.mod_exp(x2, Da, p)

            encrypted_blocks.append(x3)

        with open(output_file, 'wb') as f:
            f.write(block_size.to_bytes(4, 'big'))
            f.write(len(encrypted_blocks).to_bytes(8, 'big'))

            for block in encrypted_blocks:
                block_bytes = block.to_bytes((p.bit_length() + 7) // 8, 'big')
                f.write(len(block_bytes).to_bytes(2, 'big'))
                f.write(block_bytes)

        print(f"Файл зашифрован. Размер исходного файла: {len(data)} байт")
        print(f"Количество блоков: {len(encrypted_blocks)}")

    def decrypt_file(self, keys_a: Tuple[int, int, int], keys_b: Tuple[int, int, int],
                     input_file: str, output_file: str):
        """Дешифрование файла с помощью шифра Шамира"""
        p, Ca, Da = keys_a
        p, Cb, Db = keys_b

        with open(input_file, 'rb') as f:
            block_size = int.from_bytes(f.read(4), 'big')
            num_blocks = int.from_bytes(f.read(8), 'big')

            encrypted_blocks = []
            for _ in range(num_blocks):
                block_len = int.from_bytes(f.read(2), 'big')
                block = int.from_bytes(f.read(block_len), 'big')
                encrypted_blocks.append(block)

        decrypted_data = bytearray()

        for encrypted_block in encrypted_blocks:
            m = self.utils.mod_exp(encrypted_block, Db, p)
            decrypted_block = m.to_bytes(block_size, 'big')
            decrypted_data.extend(decrypted_block)

        decrypted_data = decrypted_data.rstrip(b'\x00')

        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

        print(f"Файл расшифрован. Размер: {len(decrypted_data)} байт")


class VernamCrypto:
    """Класс для шифра Вернама (одноразовый блокнот)"""

    def __init__(self):
        self.utils = CryptoUtils()

    def generate_key_from_dh(self, dh_key: int, key_length: int) -> bytes:
        """Генерация ключа на основе общего ключа Диффи-Хеллмана"""
        random.seed(dh_key)
        key = bytes(random.randint(0, 255) for _ in range(key_length))
        random.seed()
        return key

    def generate_random_key(self, key_length: int) -> bytes:
        """Генерация случайного ключа заданной длины"""
        return bytes(random.randint(0, 255) for _ in range(key_length))

    def encrypt_file(self, input_file: str, output_file: str, key: bytes):
        """Шифрование файла с помощью шифра Вернама"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Файл {input_file} не найден")

        file_size = os.path.getsize(input_file)
        if len(key) < file_size:
            raise ValueError(f"Длина ключа ({len(key)} байт) меньше размера файла ({file_size} байт)")

        print(f"Шифрование файла {input_file}...")
        print(f"Размер файла: {file_size} байт")
        print(f"Длина ключа: {len(key)} байт")

        with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
            # Шифруем XOR
            for i, byte in enumerate(f_in.read()):
                encrypted_byte = byte ^ key[i]
                f_out.write(bytes([encrypted_byte]))

        print(f"Файл успешно зашифрован: {output_file}")

    def decrypt_file(self, input_file: str, output_file: str, key: bytes):
        """Дешифрование файла с помощью шифра Вернама"""
        self.encrypt_file(input_file, output_file, key)
        print(f"Файл успешно расшифрован: {output_file}")

    def save_key_to_file(self, key: bytes, key_file: str):
        """Сохранение ключа в файл"""
        with open(key_file, 'wb') as f:
            f.write(key)
        print(f"Ключ сохранен в файл: {key_file}")

    def load_key_from_file(self, key_file: str) -> bytes:
        """Загрузка ключа из файла"""
        with open(key_file, 'rb') as f:
            key = f.read()
        print(f"Ключ загружен из файла: {key_file}")
        return key

    def vernam_with_dh_key_exchange(self):
        """Полный процесс: обмен ключами Диффи-Хеллмана + шифрование Вернама"""
        print("\n=== Шифр Вернама с обменом ключами Диффи-Хеллмана ===")

        print("Генерация параметров Диффи-Хеллмана...")
        p = self.utils.generate_large_prime(10 ** 8, 10 ** 9)
        g = self.utils.find_primitive_root(p)

        Xa = random.randint(2, p - 2)
        Xb = random.randint(2, p - 2)

        Ya = self.utils.mod_exp(g, Xa, p)
        Yb = self.utils.mod_exp(g, Xb, p)

        Kab_A = self.utils.mod_exp(Yb, Xa, p)
        Kab_B = self.utils.mod_exp(Ya, Xb, p)

        print(f"Параметры Диффи-Хеллмана:")
        print(f"p = {p}")
        print(f"g = {g}")
        print(f"Общий секретный ключ: {Kab_A}")

        if Kab_A != Kab_B:
            raise ValueError("Ошибка: ключи Диффи-Хеллмана не совпадают!")

        input_file = input("Введите путь к файлу для шифрования: ")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Файл {input_file} не найден")

        file_size = os.path.getsize(input_file)

        print(f"Генерация ключа Вернама длиной {file_size} байт...")
        vernam_key = self.generate_key_from_dh(Kab_A, file_size)

        output_file = input("Введите путь для сохранения зашифрованного файла: ")
        self.encrypt_file(input_file, output_file, vernam_key)

        key_file = input("Введите путь для сохранения ключа: ")
        self.save_key_to_file(vernam_key, key_file)

        print("✓ Процесс завершен успешно!")
        return output_file, key_file
