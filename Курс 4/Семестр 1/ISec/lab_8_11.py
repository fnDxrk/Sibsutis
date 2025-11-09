import hashlib
import struct
import random
import math
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
from crypto import CryptoUtils


# ============= КЛАССЫ АЛГОРИТМОВ =============


class RSASignature:
    def __init__(self):
        self.utils = CryptoUtils()

    def generate_keys(self, bit_length: int = 1024):
        print("\n=== Генерация ключей для RSA подписи ===")

        lower_bound = 2 ** (bit_length // 2 - 1)
        upper_bound = 2 ** (bit_length // 2)

        print("Генерация простых чисел p и q...")
        p = self.utils.generate_large_prime(lower_bound, upper_bound)
        q = self.utils.generate_large_prime(lower_bound, upper_bound)

        while q == p:
            q = self.utils.generate_large_prime(lower_bound, upper_bound)

        n = p * q
        phi = (p - 1) * (q - 1)

        e = 65537
        if math.gcd(e, phi) != 1:
            e = self.utils.generate_coprime(phi)

        d = self.utils.mod_inverse(e, phi)

        print(f"Сгенерированные параметры:")
        print(f"p = {p}")
        print(f"q = {q}")
        print(f"n = p * q = {n}")
        print(f"φ(n) = (p-1)*(q-1) = {phi}")
        print(f"e (открытая экспонента) = {e}")
        print(f"d (секретная экспонента) = {d}")

        public_key = (n, e)
        private_key = (n, d)

        return public_key, private_key

    def calculate_hash(self, data: bytes, hash_algorithm: str = "sha256") -> bytes:
        """Вычисление хеша данных"""
        if hash_algorithm.lower() == "sha256":
            return hashlib.sha256(data).digest()
        elif hash_algorithm.lower() == "sha512":
            return hashlib.sha512(data).digest()
        elif hash_algorithm.lower() == "sha384":
            return hashlib.sha384(data).digest()
        elif hash_algorithm.lower() == "sha1":
            return hashlib.sha1(data).digest()
        elif hash_algorithm.lower() == "md5":
            return hashlib.md5(data).digest()
        else:
            raise ValueError(f"Неподдерживаемый алгоритм хеширования: {hash_algorithm}")

    def sign_hash_byte_by_byte(
        self, hash_bytes: bytes, private_key: Tuple[int, int]
    ) -> List[int]:
        """Подписываем хеш побайтово"""
        n, d = private_key
        signature = []

        for byte in hash_bytes:
            signed_byte = self.utils.mod_exp(byte, d, n)
            signature.append(signed_byte)

        return signature

    def verify_hash_byte_by_byte(
        self, hash_bytes: bytes, signature: List[int], public_key: Tuple[int, int]
    ) -> bool:
        """Проверка хеша побайтово"""
        n, e = public_key

        if len(hash_bytes) != len(signature):
            return False

        for i, byte in enumerate(hash_bytes):
            recovered_byte = self.utils.mod_exp(signature[i], e, n) % 256

            if recovered_byte != byte:
                return False

        return True

    def sign_file(
        self,
        input_file: str,
        private_key: Tuple[int, int],
        signature_file: str = None,
        hash_algorithm: str = "sha256",
    ) -> str:
        """Создание подписи для файла"""
        print(f"\n=== Создание подписи для файла {input_file} ===")

        with open(input_file, "rb") as f:
            data = f.read()

        print(f"Размер файла: {len(data)} байт")

        file_hash = self.calculate_hash(data, hash_algorithm)
        print(f"Хеш ({hash_algorithm}): {file_hash.hex()}")

        signature = self.sign_hash_byte_by_byte(file_hash, private_key)
        print(f"Создана подпись длиной {len(signature)} элементов")

        if signature_file is None:
            signature_file = input_file + ".sig"

        self.save_signature(signature_file, signature, hash_algorithm)
        print(f"Подпись сохранена в файл: {signature_file}")

        return signature_file

    def verify_file(
        self, input_file: str, signature_file: str, public_key: Tuple[int, int]
    ) -> bool:
        """Проверка подписи файла"""
        print(f"\n=== Проверка подписи для файла {input_file} ===")

        with open(input_file, "rb") as f:
            data = f.read()

        print(f"Размер файла: {len(data)} байт")

        signature, hash_algorithm = self.load_signature(signature_file)
        print(f"Загружена подпись длиной {len(signature)} элементов")
        print(f"Алгоритм хеширования: {hash_algorithm}")

        file_hash = self.calculate_hash(data, hash_algorithm)
        print(f"Хеш ({hash_algorithm}): {file_hash.hex()}")

        is_valid = self.verify_hash_byte_by_byte(file_hash, signature, public_key)

        if is_valid:
            print("✓ Подпись ВЕРНА!")
        else:
            print("✗ Подпись НЕВЕРНА!")

        return is_valid

    def save_signature(
        self, signature_file: str, signature: List[int], hash_algorithm: str
    ):
        """Сохранение подписи в файл"""
        with open(signature_file, "wb") as f:
            alg_bytes = hash_algorithm.ljust(32).encode("utf-8")[:32]
            f.write(alg_bytes)

            f.write(struct.pack(">I", len(signature)))

            for sig_element in signature:
                byte_length = (sig_element.bit_length() + 7) // 8
                if byte_length == 0:
                    byte_length = 1

                sig_bytes = sig_element.to_bytes(byte_length, "big")
                f.write(struct.pack(">I", len(sig_bytes)))
                f.write(sig_bytes)

    def load_signature(self, signature_file: str) -> Tuple[List[int], str]:
        """Загрузка подписи из файла"""
        with open(signature_file, "rb") as f:
            alg_bytes = f.read(32)
            hash_algorithm = alg_bytes.decode("utf-8").strip()

            num_elements = struct.unpack(">I", f.read(4))[0]

            signature = []
            for _ in range(num_elements):
                elem_length = struct.unpack(">I", f.read(4))[0]

                elem_bytes = f.read(elem_length)
                sig_element = int.from_bytes(elem_bytes, "big")
                signature.append(sig_element)

        return signature, hash_algorithm

    def save_public_key(self, public_key: Tuple[int, int], key_file: str):
        """Сохранение открытого ключа"""
        n, e = public_key
        with open(key_file, "w") as f:
            f.write(f"{n}\n{e}\n")
        print(f"Открытый ключ сохранен в: {key_file}")

    def save_private_key(self, private_key: Tuple[int, int], key_file: str):
        """Сохранение закрытого ключа"""
        n, d = private_key
        with open(key_file, "w") as f:
            f.write(f"{n}\n{d}\n")
        print(f"Закрытый ключ сохранен в: {key_file}")

    def load_public_key(self, key_file: str) -> Tuple[int, int]:
        """Загрузка открытого ключа"""
        with open(key_file, "r") as f:
            n = int(f.readline().strip())
            e = int(f.readline().strip())
        print(f"Открытый ключ загружен из: {key_file}")
        return (n, e)

    def load_private_key(self, key_file: str) -> Tuple[int, int]:
        """Загрузка закрытого ключа"""
        with open(key_file, "r") as f:
            n = int(f.readline().strip())
            d = int(f.readline().strip())
        print(f"Закрытый ключ загружен из: {key_file}")
        return (n, d)


class ElGamalSignature:
    def __init__(self):
        self.utils = CryptoUtils()

    def generate_keys(
        self, key_size: int = 1024
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
        """Генерация ключей"""
        print("Генерация ключей Эль-Гамаля...")

        lower_bound = 32500
        upper_bound = 65000

        print(f"Поиск простого числа в диапазоне [{lower_bound}, {upper_bound}]...")
        p = self.utils.generate_prime(lower_bound, upper_bound)

        print(f"Найдено простое число p = {p}")

        print("Поиск примитивного корня g...")
        g = self.utils.find_primitive_root(p)
        print(f"Найден примитивный корень g = {g}")

        x = random.randint(2, p - 2)

        y = self.utils.mod_exp(g, x, p)

        public_key = (p, g, y)
        private_key = (p, x)

        print(f"Параметры сгенерированы:")
        print(f"p = {p}")
        print(f"g = {g}")
        print(f"x (секретный) = {x}")
        print(f"y (открытый) = g^x mod p = {g}^{x} mod {p} = {y}")

        print(f"\nПроверка диапазона:")
        print(f"p = {p} (в диапазоне 32500-65000)")
        print(f"Размер p: {p.bit_length()} бит")

        return public_key, private_key

    def compute_hash(self, data: bytes) -> List[int]:
        """Вычисление хеша"""
        hash_obj = hashlib.sha256()
        hash_obj.update(data)
        hash_bytes = hash_obj.digest()

        return list(hash_bytes)

    def sign_byte(self, m: int, private_key: Tuple[int, int]) -> Tuple[int, int]:
        """Подписание одного байта"""
        p, x = private_key

        while True:
            k = random.randint(2, p - 2)
            if math.gcd(k, p - 1) == 1:
                break

        r = self.utils.mod_exp(self.utils.find_primitive_root(p), k, p)

        try:
            k_inv = self.utils.mod_inverse(k, p - 1)
            s = (k_inv * (m - x * r)) % (p - 1)

            if s < 0:
                s += p - 1

        except ValueError as e:
            print(f"Ошибка при вычислении обратного элемента: {e}")
            return self.sign_byte(m, private_key)

        return r, s

    def verify_byte(
        self, m: int, signature: Tuple[int, int], public_key: Tuple[int, int, int]
    ) -> bool:
        """Проверка подписи одного байта"""
        p, g, y = public_key
        r, s = signature

        if r <= 0 or r >= p or s <= 0 or s >= p - 1:
            return False

        try:
            left_part = self.utils.mod_exp(g, m, p)
            right_part_part1 = self.utils.mod_exp(y, r, p)
            right_part_part2 = self.utils.mod_exp(r, s, p)
            right_part = (right_part_part1 * right_part_part2) % p

            return left_part == right_part
        except:
            return False

    def sign_file(
        self,
        input_file: str,
        private_key: Tuple[int, int],
        signature_file: Optional[str] = None,
    ) -> str:
        """Подписание файла"""
        print(f"Подписание файла: {input_file}")

        try:
            with open(input_file, "rb") as f:
                data = f.read()

            hash_bytes = self.compute_hash(data)
            hash_hex = "".join(f"{b:02x}" for b in hash_bytes)
            print(f"Хеш файла (SHA-256): {hash_hex}")
            print(f"Длина хеша: {len(hash_bytes)} байт")

            signatures = []
            print("Подписание байтов хеша...")
            for i, byte_val in enumerate(hash_bytes):
                r, s = self.sign_byte(byte_val, private_key)
                signatures.append((r, s))
                if i < 5:
                    print(f"  Байт {i}: значение={byte_val}, подпись=({r}, {s})")

            if signature_file is None:
                signature_file = input_file + ".sig"

            self._save_signature(signature_file, signatures, private_key[0])

            print(f"Файл успешно подписан. Подпись сохранена в: {signature_file}")
            return signature_file

        except Exception as e:
            print(f"Ошибка при подписании файла: {e}")
            raise

    def verify_file(
        self, input_file: str, signature_file: str, public_key: Tuple[int, int, int]
    ) -> bool:
        """Проверка подписи файла"""
        print(f"Проверка подписи файла: {input_file}")

        try:
            with open(input_file, "rb") as f:
                data = f.read()

            hash_bytes = self.compute_hash(data)
            hash_hex = "".join(f"{b:02x}" for b in hash_bytes)
            print(f"Хеш файла (SHA-256): {hash_hex}")

            signatures = self._load_signature(signature_file, public_key[0])
            print(f"Загружено подписей: {len(signatures)}")

            all_valid = True
            for i, byte_val in enumerate(hash_bytes):
                if i >= len(signatures):
                    print(f"Ошибка: для байта {i} нет подписи")
                    all_valid = False
                    continue

                is_valid = self.verify_byte(byte_val, signatures[i], public_key)
                if not is_valid:
                    print(f"Ошибка проверки подписи для байта {i}")
                    all_valid = False

            if all_valid:
                print("✓ Подпись ВЕРНА! Все байты прошли проверку.")
            else:
                print("✗ Подпись НЕВЕРНА!")

            return all_valid

        except Exception as e:
            print(f"Ошибка при проверке подписи: {e}")
            return False

    def _save_signature(self, filename: str, signatures: List[Tuple[int, int]], p: int):
        """Сохранение подписи в файл"""
        with open(filename, "wb") as f:
            f.write(struct.pack(">I", len(signatures)))

            for r, s in signatures:
                byte_size = (p.bit_length() + 7) // 8
                r_bytes = r.to_bytes(byte_size, "big")
                s_bytes = s.to_bytes(byte_size, "big")

                f.write(struct.pack(">I", byte_size))
                f.write(r_bytes)
                f.write(s_bytes)

    def _load_signature(self, filename: str, p: int) -> List[Tuple[int, int]]:
        """Загрузка подписи из файла"""
        signatures = []

        with open(filename, "rb") as f:
            num_sigs = struct.unpack(">I", f.read(4))[0]

            for _ in range(num_sigs):
                byte_size = struct.unpack(">I", f.read(4))[0]

                r_bytes = f.read(byte_size)
                s_bytes = f.read(byte_size)

                r = int.from_bytes(r_bytes, "big")
                s = int.from_bytes(s_bytes, "big")

                signatures.append((r, s))

        return signatures

    def save_keys(
        self,
        public_key: Tuple[int, int, int],
        private_key: Tuple[int, int],
        public_key_file: str,
        private_key_file: str,
    ):
        """Сохранение ключей"""
        p, g, y = public_key
        p_priv, x = private_key

        with open(public_key_file, "w") as f:
            f.write(f"{p}\n{g}\n{y}")

        with open(private_key_file, "w") as f:
            f.write(f"{p_priv}\n{x}")

        print(f"Открытый ключ сохранен в: {public_key_file}")
        print(f"Закрытый ключ сохранен в: {private_key_file}")

    def load_keys(
        self, public_key_file: str, private_key_file: str
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
        """Загрузка ключей"""
        with open(public_key_file, "r") as f:
            lines = f.readlines()
            p = int(lines[0].strip())
            g = int(lines[1].strip())
            y = int(lines[2].strip())

        with open(private_key_file, "r") as f:
            lines = f.readlines()
            p_priv = int(lines[0].strip())
            x = int(lines[1].strip())

        if p != p_priv:
            raise ValueError("Модули p в открытом и закрытом ключах не совпадают")

        public_key = (p, g, y)
        private_key = (p, x)

        print("Ключи успешно загружены")
        return public_key, private_key


class GOSTSignature:
    def __init__(self):
        self.utils = CryptoUtils()

    def generate_common_params(self) -> Tuple[int, int, int]:
        print("Генерация общих параметров ГОСТ Р 34.10-94...")

        print("Генерация простого числа q (16 бит)...")
        q_lower = 2**15
        q_upper = 2**16 - 1
        q = self.utils.generate_prime(q_lower, q_upper)
        print(f"q = {q} (бит: {q.bit_length()})")

        print("Генерация простого числа p (31 бит)...")
        p_lower = 2**30
        p_upper = 2**31 - 1

        while True:
            b_min = p_lower // q
            b_max = p_upper // q
            b = random.randint(b_min, b_max)
            p_candidate = b * q + 1

            if p_lower <= p_candidate <= p_upper and self.utils.test_ferma(p_candidate):
                p = p_candidate
                break

        print(f"p = {p} (бит: {p.bit_length()})")
        print(f"b = {b}")
        print(f"Проверка: p = b*q + 1 = {b}*{q} + 1 = {b * q + 1}")

        print("Генерация числа a...")
        while True:
            g = random.randint(2, p - 2)
            a = self.utils.mod_exp(g, b, p)
            if a > 1:
                check = self.utils.mod_exp(a, q, p)
                if check == 1:
                    break

        print(f"g = {g}")
        print(f"a = g^b mod p = {g}^{b} mod {p} = {a}")
        print(f"Проверка: a^q mod p = {a}^{q} mod {p} = {check}")

        return p, q, a

    def generate_keys(self, p: int, q: int, a: int) -> Tuple[int, int]:
        print("\nГенерация ключевой пары...")

        x = random.randint(1, q - 1)

        y = self.utils.mod_exp(a, x, p)

        print(f"Секретный ключ x = {x}")
        print(f"Открытый ключ y = a^x mod p = {a}^{x} mod {p} = {y}")

        return x, y

    def compute_hash(self, data: bytes, q: int) -> int:
        hash_obj = hashlib.sha256()
        hash_obj.update(data)
        hash_bytes = hash_obj.digest()

        h = int.from_bytes(hash_bytes, "big")

        h = h % (q - 1) + 1

        hash_hex = hash_obj.hexdigest()
        print(f"Хеш документа (SHA-256): {hash_hex}")
        print(f"h = {h} (0 < h < q)")

        return h

    def sign(self, data: bytes, p: int, q: int, a: int, x: int) -> Tuple[int, int]:
        print("\nПодписание документа...")

        h = self.compute_hash(data, q)

        while True:
            k = random.randint(1, q - 1)

            r = self.utils.mod_exp(a, k, p) % q
            if r == 0:
                continue

            try:
                k_inv = self.utils.mod_inverse(k, q)
                s = (k * h + x * r) % q

                if s == 0:
                    continue

                break

            except ValueError:
                continue

        print(f"Случайное число k = {k}")
        print(f"r = (a^k mod p) mod q = ({a}^{k} mod {p}) mod {q} = {r}")
        print(f"s = (k*h + x*r) mod q = ({k}*{h} + {x}*{r}) mod {q} = {s}")

        return r, s

    def verify(
        self, data: bytes, signature: Tuple[int, int], p: int, q: int, a: int, y: int
    ) -> bool:
        print("\nПроверка подписи...")

        r, s = signature

        h = self.compute_hash(data, q)

        if not (0 < r < q and 0 < s < q):
            print("Ошибка: r или s не в диапазоне (0, q)")
            return False

        print(f"Проверка: 0 < r={r} < q={q} - OK")
        print(f"Проверка: 0 < s={s} < q={q} - OK")

        try:
            h_inv = self.utils.mod_inverse(h, q)
            u1 = (s * h_inv) % q
            u2 = (-r * h_inv) % q

            print(f"h^(-1) mod q = {h_inv}")
            print(f"u1 = s * h^(-1) mod q = {s} * {h_inv} mod {q} = {u1}")
            print(f"u2 = -r * h^(-1) mod q = -{r} * {h_inv} mod {q} = {u2}")

            a_u1 = self.utils.mod_exp(a, u1, p)
            y_u2 = self.utils.mod_exp(y, u2, p)
            v = ((a_u1 * y_u2) % p) % q

            print(f"a^u1 mod p = {a}^{u1} mod {p} = {a_u1}")
            print(f"y^u2 mod p = {y}^{u2} mod {p} = {y_u2}")
            print(
                f"v = (a^u1 * y^u2 mod p) mod q = ({a_u1} * {y_u2} mod {p}) mod {q} = {v}"
            )

            print(f"Проверка: v = {v}, r = {r}")
            if v == r:
                print("✓ Подпись верна!")
                return True
            else:
                print("✗ Подпись неверна!")
                return False

        except ValueError as e:
            print(f"Ошибка при вычислении обратного элемента: {e}")
            return False

    def sign_file(
        self,
        input_file: str,
        p: int,
        q: int,
        a: int,
        x: int,
        signature_file: Optional[str] = None,
    ) -> str:
        print(f"Подписание файла: {input_file}")

        with open(input_file, "rb") as f:
            data = f.read()

        r, s = self.sign(data, p, q, a, x)

        if signature_file is None:
            signature_file = input_file + ".gost_sig"

        self._save_signature(signature_file, r, s, p.bit_length())

        print(f"Файл успешно подписан. Подпись сохранена в: {signature_file}")
        return signature_file

    def verify_file(
        self, input_file: str, signature_file: str, p: int, q: int, a: int, y: int
    ) -> bool:
        print(f"Проверка подписи файла: {input_file}")

        with open(input_file, "rb") as f:
            data = f.read()

        r, s = self._load_signature(signature_file, p.bit_length())

        return self.verify(data, (r, s), p, q, a, y)

    def _save_signature(self, filename: str, r: int, s: int, bit_length: int):
        with open(filename, "wb") as f:
            byte_size = (bit_length + 7) // 8

            r_bytes = r.to_bytes(byte_size, "big")
            s_bytes = s.to_bytes(byte_size, "big")

            f.write(struct.pack(">I", byte_size))
            f.write(r_bytes)
            f.write(s_bytes)

    def _load_signature(self, filename: str, bit_length: int) -> Tuple[int, int]:
        with open(filename, "rb") as f:
            byte_size = struct.unpack(">I", f.read(4))[0]

            r_bytes = f.read(byte_size)
            s_bytes = f.read(byte_size)

            r = int.from_bytes(r_bytes, "big")
            s = int.from_bytes(s_bytes, "big")

        return r, s

    def save_common_params(self, p: int, q: int, a: int, filename: str):
        with open(filename, "w") as f:
            f.write(f"{p}\n{q}\n{a}")

        print(f"Общие параметры сохранены в: {filename}")

    def load_common_params(self, filename: str) -> Tuple[int, int, int]:
        with open(filename, "r") as f:
            lines = f.readlines()
            p = int(lines[0].strip())
            q = int(lines[1].strip())
            a = int(lines[2].strip())

        print("Общие параметры загружены")
        return p, q, a

    def save_keys(self, x: int, y: int, private_key_file: str, public_key_file: str):
        with open(private_key_file, "w") as f:
            f.write(f"{x}")

        with open(public_key_file, "w") as f:
            f.write(f"{y}")

        print(f"Секретный ключ сохранен в: {private_key_file}")
        print(f"Открытый ключ сохранен в: {public_key_file}")

    def load_keys(self, private_key_file: str, public_key_file: str) -> Tuple[int, int]:
        with open(private_key_file, "r") as f:
            x = int(f.read().strip())

        with open(public_key_file, "r") as f:
            y = int(f.read().strip())

        print("Ключи загружены")
        return x, y


class FIPS186Signature:
    def __init__(self):
        self.utils = CryptoUtils()

    def generate_domain_parameters(
        self, L: int = 1024, N: int = 160
    ) -> Tuple[int, int, int]:
        print(f"Генерация доменных параметров DSA (L={L}, N={N})...")

        if L == 1024 and N == 160:
            print("Используются упрощенные размеры для лабораторной работы:")
            L, N = 31, 16

        print(f"Генерация простого числа q ({N} бит)...")
        q_lower = 2 ** (N - 1)
        q_upper = 2**N - 1
        q = self.utils.generate_prime(q_lower, q_upper)
        print(f"q = {q} (бит: {q.bit_length()})")

        print(f"Генерация простого числа p ({L} бит)...")
        p_lower = 2 ** (L - 1)
        p_upper = 2**L - 1

        while True:
            k_min = p_lower // q
            k_max = p_upper // q
            k = random.randint(k_min, k_max)
            p_candidate = k * q + 1

            if p_lower <= p_candidate <= p_upper and self.utils.test_ferma(p_candidate):
                p = p_candidate
                break

        print(f"p = {p} (бит: {p.bit_length()})")
        print(f"k = {k}")
        print(f"Проверка: p = k*q + 1 = {k}*{q} + 1 = {k * q + 1}")
        print(f"Проверка: (p-1) % q = {(p - 1) % q}")

        print("Генерация генератора g...")
        while True:
            h = random.randint(2, p - 2)
            g = self.utils.mod_exp(h, (p - 1) // q, p)
            if g > 1:
                break

        print(f"h = {h}")
        print(f"g = h^((p-1)/q) mod p = {h}^({(p - 1) // q}) mod {p} = {g}")

        check = self.utils.mod_exp(g, q, p)
        print(f"Проверка: g^q mod p = {g}^{q} mod {p} = {check}")

        return p, q, g

    def generate_keys(self, p: int, q: int, g: int) -> Tuple[int, int]:
        print("\nГенерация ключевой пары DSA...")

        x = random.randint(1, q - 1)

        y = self.utils.mod_exp(g, x, p)

        print(f"Секретный ключ x = {x}")
        print(f"Открытый ключ y = g^x mod p = {g}^{x} mod {p} = {y}")

        return x, y

    def compute_hash(self, data: bytes, q: int) -> int:
        hash_obj = hashlib.sha256()
        hash_obj.update(data)
        hash_bytes = hash_obj.digest()

        h = int.from_bytes(hash_bytes, "big")

        h = h % (q - 1) + 1

        hash_hex = hash_obj.hexdigest()
        print(f"Хеш документа (SHA-256): {hash_hex}")
        print(f"h = {h} (0 < h < q)")

        return h

    def sign(self, data: bytes, p: int, q: int, g: int, x: int) -> Tuple[int, int]:
        print("\nПодписание документа по DSA...")

        h = self.compute_hash(data, q)

        while True:
            k = random.randint(1, q - 1)
            r = self.utils.mod_exp(g, k, p) % q
            if r == 0:
                continue

            try:
                k_inv = self.utils.mod_inverse(k, q)
                s = (k_inv * (h + x * r)) % q

                if s == 0:
                    continue

                break

            except ValueError:
                continue

        print(f"Эфемерный ключ k = {k}")
        print(f"r = (g^k mod p) mod q = ({g}^{k} mod {p}) mod {q} = {r}")
        print(f"s = k^(-1) * (h + x*r) mod q = {k_inv} * ({h} + {x}*{r}) mod {q} = {s}")

        return r, s

    def verify(
        self, data: bytes, signature: Tuple[int, int], p: int, q: int, g: int, y: int
    ) -> bool:
        print("\nПроверка подписи DSA...")

        r, s = signature

        if not (0 < r < q and 0 < s < q):
            print("Ошибка: r или s не в диапазоне (0, q)")
            return False

        print(f"Проверка: 0 < r={r} < q={q} - OK")
        print(f"Проверка: 0 < s={s} < q={q} - OK")

        h = self.compute_hash(data, q)

        try:
            w = self.utils.mod_inverse(s, q)
            print(f"w = s^(-1) mod q = {s}^(-1) mod {q} = {w}")

            u1 = (h * w) % q
            u2 = (r * w) % q

            print(f"u1 = h * w mod q = {h} * {w} mod {q} = {u1}")
            print(f"u2 = r * w mod q = {r} * {w} mod {q} = {u2}")

            g_u1 = self.utils.mod_exp(g, u1, p)
            y_u2 = self.utils.mod_exp(y, u2, p)
            v = ((g_u1 * y_u2) % p) % q

            print(f"g^u1 mod p = {g}^{u1} mod {p} = {g_u1}")
            print(f"y^u2 mod p = {y}^{u2} mod {p} = {y_u2}")
            print(
                f"v = (g^u1 * y^u2 mod p) mod q = ({g_u1} * {y_u2} mod {p}) mod {q} = {v}"
            )

            print(f"Проверка: v = {v}, r = {r}")
            if v == r:
                print("✓ Подпись DSA верна!")
                return True
            else:
                print("✗ Подпись DSA неверна!")
                return False

        except ValueError as e:
            print(f"Ошибка при вычислении обратного элемента: {e}")
            return False

    def sign_file(
        self,
        input_file: str,
        p: int,
        q: int,
        g: int,
        x: int,
        signature_file: Optional[str] = None,
    ) -> str:
        print(f"Подписание файла: {input_file}")

        with open(input_file, "rb") as f:
            data = f.read()

        r, s = self.sign(data, p, q, g, x)

        if signature_file is None:
            signature_file = input_file + ".dsa_sig"

        self._save_signature(signature_file, r, s, p.bit_length())

        print(f"Файл успешно подписан. Подпись сохранена в: {signature_file}")
        return signature_file

    def verify_file(
        self, input_file: str, signature_file: str, p: int, q: int, g: int, y: int
    ) -> bool:
        print(f"Проверка подписи DSA файла: {input_file}")

        # Чтение файла
        with open(input_file, "rb") as f:
            data = f.read()

        r, s = self._load_signature(signature_file, p.bit_length())

        return self.verify(data, (r, s), p, q, g, y)

    def _save_signature(self, filename: str, r: int, s: int, bit_length: int):
        with open(filename, "wb") as f:
            byte_size = (bit_length + 7) // 8

            r_bytes = r.to_bytes(byte_size, "big")
            s_bytes = s.to_bytes(byte_size, "big")

            f.write(struct.pack(">I", byte_size))
            f.write(r_bytes)
            f.write(s_bytes)

    def _load_signature(self, filename: str, bit_length: int) -> Tuple[int, int]:
        with open(filename, "rb") as f:
            byte_size = struct.unpack(">I", f.read(4))[0]

            r_bytes = f.read(byte_size)
            s_bytes = f.read(byte_size)

            r = int.from_bytes(r_bytes, "big")
            s = int.from_bytes(s_bytes, "big")

        return r, s

    def save_domain_params(self, p: int, q: int, g: int, filename: str):
        with open(filename, "w") as f:
            f.write(f"{p}\n{q}\n{g}")

        print(f"Доменные параметры сохранены в: {filename}")

    def load_domain_params(self, filename: str) -> Tuple[int, int, int]:
        with open(filename, "r") as f:
            lines = f.readlines()
            p = int(lines[0].strip())
            q = int(lines[1].strip())
            g = int(lines[2].strip())

        print("Доменные параметры загружены")
        return p, q, g

    def save_keys(self, x: int, y: int, private_key_file: str, public_key_file: str):
        with open(private_key_file, "w") as f:
            f.write(f"{x}")

        with open(public_key_file, "w") as f:
            f.write(f"{y}")

        print(f"Секретный ключ сохранен в: {private_key_file}")
        print(f"Открытый ключ сохранен в: {public_key_file}")

    def load_keys(self, private_key_file: str, public_key_file: str) -> Tuple[int, int]:
        with open(private_key_file, "r") as f:
            x = int(f.read().strip())

        with open(public_key_file, "r") as f:
            y = int(f.read().strip())

        print("Ключи DSA загружены")
        return x, y


# ============= АДАПТЕРЫ =============


class SignatureAlgorithm(ABC):
    """Базовый интерфейс для всех алгоритмов подписи"""

    @abstractmethod
    def generate_keys(self, **kwargs):
        pass

    @abstractmethod
    def sign_file(
        self,
        input_file: str,
        private_key,
        signature_file: Optional[str] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def verify_file(self, input_file: str, signature_file: str, public_key) -> bool:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class RSAAdapter(SignatureAlgorithm):
    """Адаптер для RSA"""

    def __init__(self):
        self.rsa = RSASignature()

    def generate_keys(self, **kwargs):
        bit_length = kwargs.get("bit_length", 1024)
        return self.rsa.generate_keys(bit_length)

    def sign_file(
        self,
        input_file: str,
        private_key,
        signature_file: Optional[str] = None,
        **kwargs,
    ) -> str:
        hash_algorithm = kwargs.get("hash_algorithm", "sha256")
        return self.rsa.sign_file(
            input_file, private_key, signature_file, hash_algorithm
        )

    def verify_file(self, input_file: str, signature_file: str, public_key) -> bool:
        return self.rsa.verify_file(input_file, signature_file, public_key)

    def get_name(self) -> str:
        return "RSA"


class ElGamalAdapter(SignatureAlgorithm):
    """Адаптер для ElGamal"""

    def __init__(self):
        self.elgamal = ElGamalSignature()

    def generate_keys(self, **kwargs):
        key_size = kwargs.get("key_size", 1024)
        return self.elgamal.generate_keys(key_size)

    def sign_file(
        self,
        input_file: str,
        private_key,
        signature_file: Optional[str] = None,
        **kwargs,
    ) -> str:
        return self.elgamal.sign_file(input_file, private_key, signature_file)

    def verify_file(self, input_file: str, signature_file: str, public_key) -> bool:
        return self.elgamal.verify_file(input_file, signature_file, public_key)

    def get_name(self) -> str:
        return "ElGamal"


class GOSTAdapter(SignatureAlgorithm):
    """Адаптер для ГОСТ Р 34.10-94"""

    def __init__(self):
        self.gost = GOSTSignature()
        self.common_params = None  # (p, q, a)

    def generate_keys(self, **kwargs):
        # Сначала генерируем общие параметры
        self.common_params = self.gost.generate_common_params()
        p, q, a = self.common_params
        # Затем генерируем ключевую пару
        x, y = self.gost.generate_keys(p, q, a)
        # Для ГОСТ public_key = (p, q, a, y), private_key = (p, q, a, x)
        public_key = (p, q, a, y)
        private_key = (p, q, a, x)
        return public_key, private_key

    def sign_file(
        self,
        input_file: str,
        private_key,
        signature_file: Optional[str] = None,
        **kwargs,
    ) -> str:
        p, q, a, x = private_key
        return self.gost.sign_file(input_file, p, q, a, x, signature_file)

    def verify_file(self, input_file: str, signature_file: str, public_key) -> bool:
        p, q, a, y = public_key
        return self.gost.verify_file(input_file, signature_file, p, q, a, y)

    def get_name(self) -> str:
        return "ГОСТ Р 34.10-94"


class FIPS186Adapter(SignatureAlgorithm):
    """Адаптер для FIPS 186 (DSA)"""

    def __init__(self):
        self.dsa = FIPS186Signature()
        self.domain_params = None  # (p, q, g)

    def generate_keys(self, **kwargs):
        # Сначала генерируем доменные параметры
        L = kwargs.get("L", 1024)
        N = kwargs.get("N", 160)
        self.domain_params = self.dsa.generate_domain_parameters(L, N)
        p, q, g = self.domain_params
        # Затем генерируем ключевую пару
        x, y = self.dsa.generate_keys(p, q, g)
        # Для DSA public_key = (p, q, g, y), private_key = (p, q, g, x)
        public_key = (p, q, g, y)
        private_key = (p, q, g, x)
        return public_key, private_key

    def sign_file(
        self,
        input_file: str,
        private_key,
        signature_file: Optional[str] = None,
        **kwargs,
    ) -> str:
        p, q, g, x = private_key
        return self.dsa.sign_file(input_file, p, q, g, x, signature_file)

    def verify_file(self, input_file: str, signature_file: str, public_key) -> bool:
        p, q, g, y = public_key
        return self.dsa.verify_file(input_file, signature_file, p, q, g, y)

    def get_name(self) -> str:
        return "FIPS 186 (DSA)"


# ============= МЕНЮ =============


def unified_menu(algorithm: SignatureAlgorithm):
    """Единое меню для всех алгоритмов"""
    public_key = None
    private_key = None

    while True:
        print(f"\n{'=' * 50}")
        print(f"   Электронная подпись: {algorithm.get_name()}")
        print(f"{'=' * 50}")
        print("1. Генерация ключей")
        print("2. Создание подписи для файла")
        print("3. Проверка подписи файла")
        print("0. Назад")

        choice = input("Выберите действие: ").strip()

        if choice == "1":
            try:
                params = {}
                # Запрашиваем параметры для разных алгоритмов
                if algorithm.get_name() == "RSA":
                    bit_length = input(
                        "Длина ключа в битах (по умолчанию 1024): "
                    ).strip()
                    if bit_length:
                        params["bit_length"] = int(bit_length)

                elif algorithm.get_name() == "ElGamal":
                    key_size = input("Размер ключа (по умолчанию 1024): ").strip()
                    if key_size:
                        params["key_size"] = int(key_size)

                elif algorithm.get_name() == "FIPS 186 (DSA)":
                    L = input("Длина p в битах (по умолчанию 1024): ").strip()
                    N = input("Длина q в битах (по умолчанию 160): ").strip()
                    if L:
                        params["L"] = int(L)
                    if N:
                        params["N"] = int(N)

                # Для ГОСТ параметры не запрашиваем - генерируются автоматически

                public_key, private_key = algorithm.generate_keys(**params)
                print("✓ Ключи успешно сгенерированы!")

            except Exception as e:
                print(f"✗ Ошибка генерации: {e}")

        elif choice == "2":
            if private_key is None:
                print("⚠ Сначала сгенерируйте ключи!")
                continue

            input_file = input("Путь к файлу для подписи: ").strip()
            sig_file = (
                input("Файл подписи (Enter для автоматического): ").strip() or None
            )

            params = {}
            # Запрашиваем алгоритм хеширования только для RSA
            if algorithm.get_name() == "RSA":
                hash_alg = input(
                    "Алгоритм хеширования (sha256/sha512/sha384/sha1/md5, по умолчанию sha256): "
                ).strip()
                if hash_alg:
                    params["hash_algorithm"] = hash_alg

            try:
                result = algorithm.sign_file(
                    input_file, private_key, sig_file, **params
                )
                print(f"✓ Файл успешно подписан: {result}")
            except Exception as e:
                print(f"✗ Ошибка подписания: {e}")

        elif choice == "3":
            if public_key is None:
                print("⚠ Сначала сгенерируйте ключи!")
                continue

            input_file = input("Путь к файлу для проверки: ").strip()
            sig_file = input("Файл подписи: ").strip()

            try:
                is_valid = algorithm.verify_file(input_file, sig_file, public_key)
                if is_valid:
                    print("✓ Подпись ДЕЙСТВИТЕЛЬНА!")
                else:
                    print("✗ Подпись НЕДЕЙСТВИТЕЛЬНА!")
            except Exception as e:
                print(f"✗ Ошибка проверки: {e}")

        elif choice == "0":
            break

        else:
            print("✗ Неверный выбор!")


def main():
    algorithms = {
        "1": RSAAdapter(),
        "2": ElGamalAdapter(),
        "3": GOSTAdapter(),
        "4": FIPS186Adapter(),
    }

    while True:
        print(f"\n{'=' * 50}")
        print("   СИСТЕМА ЭЛЕКТРОННОЙ ПОДПИСИ")
        print(f"{'=' * 50}")
        print("1. RSA")
        print("2. ElGamal")
        print("3. ГОСТ Р 34.10-94")
        print("4. FIPS 186 (DSA)")
        print("0. Выход")

        choice = input("Выберите алгоритм: ").strip()

        if choice == "0":
            print("Выход из программы")
            break

        if choice in algorithms:
            unified_menu(algorithms[choice])
        else:
            print("✗ Неверный выбор!")


if __name__ == "__main__":
    main()
