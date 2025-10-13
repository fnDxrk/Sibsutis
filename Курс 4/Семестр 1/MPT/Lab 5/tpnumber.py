import math


class TPNumberException(Exception):
    """Класс для исключительных ситуаций TPNumber"""
    pass


class TPNumber:
    def __init__(self, *args):
        """
        Конструкторы:
        1. TPNumber(a: float, b: int, c: int)
        2. TPNumber(a_str: str, b_str: str, c_str: str)
        """
        if len(args) == 3:
            if isinstance(args[0], (int, float)) and isinstance(args[1], int) and isinstance(args[2], int):
                self._init_from_numbers(args[0], args[1], args[2])
            elif isinstance(args[0], str) and isinstance(args[1], str) and isinstance(args[2], str):
                self._init_from_strings(args[0], args[1], args[2])
            else:
                raise TPNumberException("Неверные типы аргументов")
        else:
            raise TPNumberException("Неверное количество аргументов")

    def _init_from_numbers(self, a: float, b: int, c: int):
        """Конструктор из чисел"""
        if not (2 <= b <= 16):
            raise TPNumberException("Основание системы счисления должно быть в диапазоне [2..16]")
        if c < 0:
            raise TPNumberException("Точность представления должна быть >= 0")
        self._n = float(a)
        self._b = b
        self._c = c

    def _init_from_strings(self, a_str: str, b_str: str, c_str: str):
        """Конструктор из строк"""
        try:
            b = int(b_str)
            c = int(c_str)
        except ValueError:
            raise TPNumberException("Основание и точность должны быть целыми числами")

        if not (2 <= b <= 16):
            raise TPNumberException("Основание системы счисления должно быть в диапазоне [2..16]")
        if c < 0:
            raise TPNumberException("Точность представления должна быть >= 0")

        # Преобразование строки в число
        try:
            n = self._convert_string_to_number(a_str, b)
        except ValueError:
            raise TPNumberException(f"Невозможно преобразовать строку '{a_str}' в число с основанием {b}")

        self._n = n
        self._b = b
        self._c = c

    def _convert_string_to_number(self, s: str, base: int) -> float:
        """Преобразование строки в число с заданным основанием"""
        s = s.strip().lower()

        # Проверка на отрицательное число
        negative = False
        if s.startswith('-'):
            negative = True
            s = s[1:]

        # Разделение на целую и дробную части
        parts = s.split('.')
        if len(parts) > 2:
            raise ValueError("Неверный формат числа")

        integer_part = parts[0]
        fractional_part = parts[1] if len(parts) > 1 else ""

        # Преобразование целой части
        integer_value = 0
        for char in integer_part:
            digit = self._char_to_digit(char)
            if digit >= base:
                raise ValueError(f"Цифра {char} недопустима для основания {base}")
            integer_value = integer_value * base + digit

        # Преобразование дробной части
        fractional_value = 0.0
        if fractional_part:
            for i, char in enumerate(fractional_part, 1):
                digit = self._char_to_digit(char)
                if digit >= base:
                    raise ValueError(f"Цифра {char} недопустима для основания {base}")
                fractional_value += digit / (base ** i)

        result = integer_value + fractional_value
        return -result if negative else result

    def _char_to_digit(self, char: str) -> int:
        """Преобразование символа в цифру"""
        if '0' <= char <= '9':
            return ord(char) - ord('0')
        elif 'a' <= char <= 'f':
            return 10 + ord(char) - ord('a')
        else:
            raise ValueError(f"Недопустимый символ: {char}")

    def _digit_to_char(self, digit: int) -> str:
        """Преобразование цифры в символ"""
        if 0 <= digit <= 9:
            return str(digit)
        elif 10 <= digit <= 15:
            return chr(ord('a') + digit - 10)
        else:
            raise ValueError(f"Недопустимая цифра: {digit}")

    def copy(self):
        """Создать копию числа"""
        return TPNumber(self._n, self._b, self._c)

    def add(self, other):
        """Сложение"""
        if not isinstance(other, TPNumber):
            raise TPNumberException("Можно складывать только с TPNumber")
        if self._b != other._b or self._c != other._c:
            raise TPNumberException("Основания и точности должны совпадать")
        return TPNumber(self._n + other._n, self._b, self._c)

    def multiply(self, other):
        """Умножение"""
        if not isinstance(other, TPNumber):
            raise TPNumberException("Можно умножать только на TPNumber")
        if self._b != other._b or self._c != other._c:
            raise TPNumberException("Основания и точности должны совпадать")
        return TPNumber(self._n * other._n, self._b, self._c)

    def subtract(self, other):
        """Вычитание"""
        if not isinstance(other, TPNumber):
            raise TPNumberException("Можно вычитать только TPNumber")
        if self._b != other._b or self._c != other._c:
            raise TPNumberException("Основания и точности должны совпадать")
        return TPNumber(self._n - other._n, self._b, self._c)

    def divide(self, other):
        """Деление"""
        if not isinstance(other, TPNumber):
            raise TPNumberException("Можно делить только на TPNumber")
        if self._b != other._b or self._c != other._c:
            raise TPNumberException("Основания и точности должны совпадать")
        if other._n == 0:
            raise TPNumberException("Деление на ноль невозможно")
        return TPNumber(self._n / other._n, self._b, self._c)

    def invert(self):
        """Обратное число (1/n)"""
        if self._n == 0:
            raise TPNumberException("Невозможно вычислить обратное число для нуля")
        return TPNumber(1.0 / self._n, self._b, self._c)

    def square(self):
        """Квадрат числа"""
        return TPNumber(self._n * self._n, self._b, self._c)

    def get_number(self) -> float:
        """Получить число как вещественное значение"""
        return self._n

    def get_number_string(self) -> str:
        """Получить число как строку в системе счисления с основанием b"""
        return self._convert_number_to_string(self._n, self._b, self._c)

    def _convert_number_to_string(self, number: float, base: int, precision: int) -> str:
        """Преобразование числа в строку с заданным основанием и точностью"""
        if number == 0:
            return "0" + (("." + "0" * precision) if precision > 0 else "")

        negative = number < 0
        number = abs(number)

        # Целая часть
        integer_part = int(number)
        fractional_part = number - integer_part

        # Преобразование целой части
        integer_str = ""
        if integer_part == 0:
            integer_str = "0"
        else:
            temp = integer_part
            while temp > 0:
                digit = temp % base
                integer_str = self._digit_to_char(digit) + integer_str
                temp //= base

        # Преобразование дробной части с дополнением нулями
        fractional_str = ""
        if precision > 0:
            fractional_str = "."
            temp = fractional_part
            for _ in range(precision):
                temp *= base
                digit = int(temp)
                fractional_str += self._digit_to_char(digit)
                temp -= digit

        result = integer_str + fractional_str
        return ("-" + result) if negative else result

    def get_base_number(self) -> int:
        """Получить основание как число"""
        return self._b

    def get_base_string(self) -> str:
        """Получить основание как строку"""
        return str(self._b)

    def get_precision_number(self) -> int:
        """Получить точность как число"""
        return self._c

    def get_precision_string(self) -> str:
        """Получить точность как строку"""
        return str(self._c)

    def set_base_number(self, new_b: int):
        """Установить основание (число)"""
        if not (2 <= new_b <= 16):
            raise TPNumberException("Основание системы счисления должно быть в диапазоне [2..16]")
        self._b = new_b

    def set_base_string(self, bs: str):
        """Установить основание (строка)"""
        try:
            new_b = int(bs)
        except ValueError:
            raise TPNumberException("Основание должно быть целым числом")
        if not (2 <= new_b <= 16):
            raise TPNumberException("Основание системы счисления должно быть в диапазоне [2..16]")
        self._b = new_b

    def set_precision_number(self, new_c: int):
        """Установить точность (число)"""
        if new_c < 0:
            raise TPNumberException("Точность представления должна быть >= 0")
        self._c = new_c

    def set_precision_string(self, new_c: str):
        """Установить точность (строка)"""
        try:
            c = int(new_c)
        except ValueError:
            raise TPNumberException("Точность должна быть целым числом")
        if c < 0:
            raise TPNumberException("Точность представления должна быть >= 0")
        self._c = c

    def __str__(self):
        return f"TPNumber({self.get_number_string()}, base={self._b}, precision={self._c})"

    def __repr__(self):
        return self.__str__()

