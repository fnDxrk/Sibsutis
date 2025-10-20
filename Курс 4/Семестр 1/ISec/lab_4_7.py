import os
from crypto import CryptoUtils, RSACrypto, DiffieHellman, ElGamalCrypto, ShamirCrypto, VernamCrypto


class CryptoSystem:
    """Основной класс системы с меню"""

    def __init__(self):
        self.utils = CryptoUtils()
        self.rsa = RSACrypto()
        self.dh = DiffieHellman()
        self.elgamal = ElGamalCrypto()
        self.shamir = ShamirCrypto()
        self.vernam = VernamCrypto()

    def create_test_file(self, filename: str = "test_file.txt") -> str:
        """Создание тестового файла"""
        test_content = """Hello, World!"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"Создан тестовый файл: {filename}")
        return filename

    def test_encryption_systems(self):
        """Тестирование систем шифрования"""
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ СИСТЕМ ШИФРОВАНИЯ")
        print("=" * 60)

        test_file = self.create_test_file()

        print("\n1. ТЕСТ ШИФРА ЭЛЬ-ГАМАЛЯ")
        print("-" * 40)

        public_key, private_key = self.elgamal.generate_keys(2)
        self.elgamal.encrypt_file(public_key, test_file, "encrypted_elgamal.bin")
        self.elgamal.decrypt_file(private_key, "encrypted_elgamal.bin", "decrypted_elgamal.txt")

        with open(test_file, 'r', encoding='utf-8') as f1, open("decrypted_elgamal.txt", 'r', encoding='utf-8') as f2:
            original = f1.read()
            decrypted = f2.read()

        if original == decrypted:
            print("✓ Шифр Эль-Гамаля: УСПЕХ")
            print(f"Исходный размер: {len(original)} символов")
            print(f"Расшифрованный размер: {len(decrypted)} символов")
        else:
            print("✗ Шифр Эль-Гамаля: ОШИБКА")

        print("\n2. ТЕСТ ШИФРА ШАМИРА")
        print("-" * 40)

        keys_a, keys_b = self.shamir.generate_keys(2)
        self.shamir.encrypt_file(keys_a, keys_b, test_file, "encrypted_shamir.bin")
        self.shamir.decrypt_file(keys_a, keys_b, "encrypted_shamir.bin", "decrypted_shamir.txt")

        with open(test_file, 'r', encoding='utf-8') as f1, open("decrypted_shamir.txt", 'r', encoding='utf-8') as f2:
            original = f1.read()
            decrypted = f2.read()

        if original == decrypted:
            print("✓ Шифр Шамира: УСПЕХ")
            print(f"Исходный размер: {len(original)} символов")
            print(f"Расшифрованный размер: {len(decrypted)} символов")
        else:
            print("✗ Шифр Шамира: ОШИБКА")

        print("\n3. ТЕСТ RSA ШИФРОВАНИЯ")
        print("-" * 40)

        public_key, private_key = self.rsa.generate_keys(2)
        self.rsa.encrypt_file(public_key, test_file, "encrypted_rsa.bin")
        self.rsa.decrypt_file(private_key, "encrypted_rsa.bin", "decrypted_rsa.txt")

        with open(test_file, 'r', encoding='utf-8') as f1, open("decrypted_rsa.txt", 'r', encoding='utf-8') as f2:
            original = f1.read()
            decrypted = f2.read()

        if original == decrypted:
            print("✓ RSA Шифрование: УСПЕХ")
            print(f"Исходный размер: {len(original)} символов")
            print(f"Расшифрованный размер: {len(decrypted)} символов")
        else:
            print("✗ RSA Шифрование: ОШИБКА")

        print("\n4. ТЕСТ ШИФРА ВЕРНАМА")
        print("-" * 40)

        try:
            with open(test_file, 'rb') as f:
                original_data = f.read()

            key = self.vernam.generate_random_key(len(original_data))

            self.vernam.encrypt_file(test_file, "encrypted_vernam.bin", key)

            self.vernam.save_key_to_file(key, "vernam_key.bin")

            loaded_key = self.vernam.load_key_from_file("vernam_key.bin")
            self.vernam.decrypt_file("encrypted_vernam.bin", "decrypted_vernam.txt", loaded_key)

            with open("decrypted_vernam.txt", 'rb') as f:
                decrypted_data = f.read()

            if original_data == decrypted_data:
                print("✓ Шифр Вернама: УСПЕХ")
                print(f"Исходный размер: {len(original_data)} байт")
                print(f"Расшифрованный размер: {len(decrypted_data)} байт")
            else:
                print("✗ Шифр Вернама: ОШИБКА")

        except Exception as e:
            print(f"✗ Шифр Вернама: ОШИБКА - {e}")

        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 60)


    def rsa_crypto_system(self):
        """Основная функция для работы с RSA"""
        print("\n=== RSA Шифрование ===")

        print("Выберите действие:")
        print("1 - Шифрование файла")
        print("2 - Дешифрование файла")
        action = int(input("Ваш выбор: "))

        print("\nВыберите способ задания параметров:")
        print("1 - Ввести параметры вручную")
        print("2 - Сгенерировать параметры автоматически")
        mode = int(input("Ваш выбор: "))

        if action == 1:
            public_key, private_key = self.rsa.generate_keys(mode)

            input_file = input("Введите путь к файлу для шифрования: ")
            output_file = input("Введите путь для сохранения зашифрованного файла: ")

            if not os.path.exists(input_file):
                print("Ошибка: исходный файл не существует!")
                return

            print("Шифрование...")
            self.rsa.encrypt_file(public_key, input_file, output_file)

            print(f"✓ Файл успешно зашифрован и сохранен как {output_file}")
            print(f"Секретный ключ для дешифрования: d = {private_key[1]}")
            print(f"Параметры p и q: p = {private_key[2]}, q = {private_key[3]}")

        elif action == 2:
            if mode == 1:
                n = int(input("Введите модуль n: "))
                d = int(input("Введите секретный ключ d: "))
                p = int(input("Введите простое число p: "))
                q = int(input("Введите простое число q: "))
            else:
                print("Для дешифрования необходимо знать секретный ключ!")
                return

            private_key = (n, d, p, q)

            input_file = input("Введите путь к зашифрованному файлу: ")
            output_file = input("Введите путь для сохранения расшифрованного файла: ")

            if not os.path.exists(input_file):
                print("Ошибка: зашифрованный файл не существует!")
                return

            print("Дешифрование...")
            self.rsa.decrypt_file(private_key, input_file, output_file)

            print(f"✓ Файл успешно расшифрован и сохранен как {output_file}")

            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("\nСодержимое файла (как текст):")
                    print(content[:500] + "..." if len(content) > 500 else content)
            except:
                print("\nФайл содержит бинарные данные (не текст)")

        else:
            print("Неверный выбор действия!")

    def elgamal_crypto_system(self):
        """Основная функция для работы с шифром Эль-Гамаля"""
        print("\n=== Шифр Эль-Гамаля ===")

        print("Выберите действие:")
        print("1 - Шифрование файла")
        print("2 - Дешифрование файла")
        action = int(input("Ваш выбор: "))

        print("\nВыберите способ задания параметров:")
        print("1 - Ввести параметры вручную")
        print("2 - Сгенерировать параметры автоматически")
        mode = int(input("Ваш выбор: "))

        if action == 1:
            public_key, private_key = self.elgamal.generate_keys(mode)

            input_file = input("Введите путь к файлу для шифрования: ")
            output_file = input("Введите путь для сохранения зашифрованного файла: ")

            if not os.path.exists(input_file):
                print("Ошибка: исходный файл не существует!")
                return

            print("Шифрование...")
            self.elgamal.encrypt_file(public_key, input_file, output_file)

            print(f"✓ Файл успешно зашифрован и сохранен как {output_file}")
            print(f"Секретный ключ для дешифрования: D1 = {private_key[1]}")

        elif action == 2:
            if mode == 1:
                p = int(input("Введите простое число p: "))
                D1 = int(input("Введите секретный ключ D1: "))
            else:
                print("Для дешифрования необходимо знать секретный ключ!")
                return

            private_key = (p, D1)

            input_file = input("Введите путь к зашифрованному файлу: ")
            output_file = input("Введите путь для сохранения расшифрованного файла: ")

            if not os.path.exists(input_file):
                print("Ошибка: зашифрованный файл не существует!")
                return

            print("Дешифрование...")
            self.elgamal.decrypt_file(private_key, input_file, output_file)

            print(f"✓ Файл успешно расшифрован и сохранен как {output_file}")

            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("\nСодержимое файла (как текст):")
                    print(content[:500] + "..." if len(content) > 500 else content)
            except:
                print("\nФайл содержит бинарные данные (не текст)")

        else:
            print("Неверный выбор действия!")

    def shamir_crypto_system(self):
        """Основная функция для работы с шифром Шамира"""
        print("\n=== Шифр Шамира ===")

        print("Выберите действие:")
        print("1 - Шифрование файла")
        print("2 - Дешифрование файла")
        action = int(input("Ваш выбор: "))

        print("\nВыберите способ задания параметров:")
        print("1 - Ввести параметры вручную")
        print("2 - Сгенерировать параметры автоматически")
        mode = int(input("Ваш выбор: "))

        if action == 1:
            keys_a, keys_b = self.shamir.generate_keys(mode)

            input_file = input("Введите путь к файлу для шифрования: ")
            output_file = input("Введите путь для сохранения зашифрованного файла: ")

            if not os.path.exists(input_file):
                print("Ошибка: исходный файл не существует!")
                return

            print("Шифрование...")
            self.shamir.encrypt_file(keys_a, keys_b, input_file, output_file)

            print(f"✓ Файл успешно зашифрован и сохранен как {output_file}")
            print(f"Секретные ключи для дешифрования:")
            print(f"Da = {keys_a[2]}")
            print(f"Db = {keys_b[2]}")

        elif action == 2:
            if mode == 1:
                p = int(input("Введите простое число p: "))
                Ca = int(input("Введите открытый ключ абонента A (Ca): "))
                Da = int(input("Введите секретный ключ абонента A (Da): "))
                Cb = int(input("Введите открытый ключ абонента B (Cb): "))
                Db = int(input("Введите секретный ключ абонента B (Db): "))
            else:
                print("Для дешифрования необходимо знать секретные ключи!")
                return

            keys_a = (p, Ca, Da)
            keys_b = (p, Cb, Db)

            input_file = input("Введите путь к зашифрованному файлу: ")
            output_file = input("Введите путь для сохранения расшифрованного файла: ")

            if not os.path.exists(input_file):
                print("Ошибка: зашифрованный файл не существует!")
                return

            print("Дешифрование...")
            self.shamir.decrypt_file(keys_a, keys_b, input_file, output_file)

            print(f"✓ Файл успешно расшифрован и сохранен как {output_file}")

            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("\nСодержимое файла (как текст):")
                    print(content[:500] + "..." if len(content) > 500 else content)
            except:
                print("\nФайл содержит бинарные данные (не текст)")

        else:
            print("Неверный выбор действия!")

    def vernam_crypto_system(self):
        """Основная функция для работы с шифром Вернама"""
        print("\n=== Шифр Вернама (одноразовый блокнот) ===")

        print("Выберите действие:")
        print("1 - Шифрование файла")
        print("2 - Дешифрование файла")
        print("3 - Полный процесс с обменом ключами Диффи-Хеллмана")
        action = int(input("Ваш выбор: "))

        try:
            if action == 1:
                self._vernam_encrypt()
            elif action == 2:
                self._vernam_decrypt()
            elif action == 3:
                self._vernam_with_dh()
            else:
                print("Неверный выбор действия!")
        except Exception as e:
            print(f"Ошибка: {e}")

    def _vernam_encrypt(self):
        """Шифрование файла с помощью Вернама"""
        input_file = input("Введите путь к файлу для шифрования: ")
        output_file = input("Введите путь для сохранения зашифрованного файла: ")

        if not os.path.exists(input_file):
            print("Ошибка: исходный файл не существует!")
            return

        file_size = os.path.getsize(input_file)

        print("\nВыберите способ генерации ключа:")
        print("1 - Сгенерировать случайный ключ")
        print("2 - Использовать существующий ключ")
        key_choice = int(input("Ваш выбор: "))

        if key_choice == 1:
            # Генерация случайного ключа
            key = self.vernam.generate_random_key(file_size)
            key_file = input("Введите путь для сохранения ключа: ")
            self.vernam.save_key_to_file(key, key_file)
        elif key_choice == 2:
            # Загрузка существующего ключа
            key_file = input("Введите путь к файлу с ключом: ")
            key = self.vernam.load_key_from_file(key_file)
        else:
            print("Неверный выбор!")
            return

        # Шифрование
        self.vernam.encrypt_file(input_file, output_file, key)
        print(f"✓ Файл успешно зашифрован: {output_file}")

    def _vernam_decrypt(self):
        """Дешифрование файла с помощью Вернама"""
        input_file = input("Введите путь к зашифрованному файлу: ")
        output_file = input("Введите путь для сохранения расшифрованного файла: ")
        key_file = input("Введите путь к файлу с ключом: ")

        if not os.path.exists(input_file):
            print("Ошибка: зашифрованный файл не существует!")
            return
        if not os.path.exists(key_file):
            print("Ошибка: файл с ключом не существует!")
            return

        # Загрузка ключа и дешифрование
        key = self.vernam.load_key_from_file(key_file)
        self.vernam.decrypt_file(input_file, output_file, key)

        # Попытка показать содержимое текстового файла
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("\nСодержимое файла (первые 500 символов):")
                print(content[:500] + "..." if len(content) > 500 else content)
        except:
            print("\nФайл содержит бинарные данные (не текст)")

    def _vernam_with_dh(self):
        """Полный процесс с Диффи-Хеллманом"""
        try:
            output_file, key_file = self.vernam.vernam_with_dh_key_exchange()
            print(f"\nРезультаты:")
            print(f"Зашифрованный файл: {output_file}")
            print(f"Ключ: {key_file}")
            print("\nДля дешифрования используйте опцию 2 в меню Вернама")
        except Exception as e:
            print(f"Ошибка в процессе: {e}")


    def main(self):
        """Основное меню программы"""
        while True:
            print("1 - Схема Диффи-Хеллмана (общий ключ)")
            print("2 - Шифр Эль-Гамаля (шифрование/дешифрование файлов)")
            print("3 - Шифр Шамира (шифрование/дешифрование файлов)")
            print("4 - RSA (шифрование/дешифрование файлов)")
            print("5 - Шифр Вернама (одноразовый блокнот)")
            print("6 - Тестирование систем шифрования")
            print("0 - Выход")

            try:
                choice = int(input("Ваш выбор: "))
            except ValueError:
                print("Ошибка: введите число от 0 до 6")
                continue

            if choice == 0:
                print("Выход из программы...")
                break
            elif choice == 1:
                self.dh.key_exchange()
            elif choice == 2:
                self.elgamal_crypto_system()
            elif choice == 3:
                self.shamir_crypto_system()
            elif choice == 4:
                self.rsa_crypto_system()
            elif choice == 5:
                self.vernam_crypto_system()
            elif choice == 6:
                self.test_encryption_systems()
            else:
                print("Неверный выбор! Попробуйте снова.")

            input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    crypto_system = CryptoSystem()
    crypto_system.main()
