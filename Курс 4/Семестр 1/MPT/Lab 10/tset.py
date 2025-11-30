from typing import TypeVar, Generic, Iterable, Iterator

T = TypeVar("T")


class TFrac:
    """Класс для представления простой дроби"""

    def __init__(self, numerator=0, denominator=1):
        """
        Инициализация дроби
        Args:
            numerator: числитель
            denominator: знаменатель (не может быть 0)
        Raises:
            ValueError: если знаменатель равен 0
        """
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        self.numerator = numerator
        self.denominator = denominator
        self._normalize()

    def _normalize(self):
        """Приведение дроби к нормальной форме (сокращение)"""
        if self.numerator == 0:
            self.denominator = 1
            return

        a, b = abs(self.numerator), abs(self.denominator)
        while b:
            a, b = b, a % b
        gcd = a

        self.numerator //= gcd
        self.denominator //= gcd

        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator

    def __eq__(self, other):
        """Проверка равенства дробей"""
        if isinstance(other, TFrac):
            return (
                self.numerator * other.denominator == other.numerator * self.denominator
            )
        return False

    def __hash__(self):
        """Хеш для использования в множествах"""
        return hash((self.numerator, self.denominator))

    def __str__(self):
        """Строковое представление дроби"""
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self):
        """Представление для отладки"""
        return f"TFrac({self.numerator}, {self.denominator})"


class TSet(Generic[T], set):
    """
    Универсальное множество элементов типа T
    Наследуется от встроенного типа set с добавлением типизации и дополнительных методов
    """

    def __init__(self, elements: Iterable[T] = None):
        """
        Конструктор множества
        Args:
            elements: итерируемый объект с элементами (опционально)
        """
        if elements is None:
            super().__init__()
        else:
            super().__init__(elements)

    def clear(self) -> None:
        """
        Опустошить множество - удалить все элементы
        """
        super().clear()

    def add(self, element: T) -> None:
        """
        Добавить элемент в множество, если его там нет
        Args:
            element: элемент для добавления
        """
        super().add(element)

    def remove(self, element: T) -> None:
        """
        Удалить элемент из множества, если он присутствует
        Args:
            element: элемент для удаления
        Raises:
            KeyError: если элемент не найден в множестве
        """
        super().remove(element)

    def is_empty(self) -> bool:
        """
        Проверить, пусто ли множество
        Returns:
            True если множество пустое, иначе False
        """
        return len(self) == 0

    def contains(self, element: T) -> bool:
        """
        Проверить принадлежность элемента множеству
        Args:
            element: элемент для проверки
        Returns:
            True если элемент принадлежит множеству, иначе False
        """
        return element in self

    def union(self, other: "TSet[T]") -> "TSet[T]":
        """
        Объединить с другим множеством
        Args:
            other: другое множество для объединения
        Returns:
            Новое множество - объединение текущего с другим
        """
        if not isinstance(other, TSet):
            raise TypeError("Argument must be TSet")
        return TSet(super().__or__(other))

    def difference(self, other: "TSet[T]") -> "TSet[T]":
        """
        Вычесть другое множество
        Args:
            other: множество для вычитания
        Returns:
            Новое множество - разность текущего с другим
        """
        if not isinstance(other, TSet):
            raise TypeError("Argument must be TSet")
        return TSet(super().__sub__(other))

    def intersection(self, other: "TSet[T]") -> "TSet[T]":
        """
        Пересечь с другим множеством
        Args:
            other: множество для пересечения
        Returns:
            Новое множество - пересечение текущего с другим
        """
        if not isinstance(other, TSet):
            raise TypeError("Argument must be TSet")
        return TSet(super().__and__(other))

    def size(self) -> int:
        """
        Получить количество элементов в множестве
        Returns:
            Количество элементов
        """
        return len(self)

    def get_element(self, index: int) -> T:
        """
        Получить элемент по индексу (для итерации)
        Args:
            index: индекс элемента
        Returns:
            Элемент по указанному индексу
        Raises:
            IndexError: если индекс вне диапазона
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        return list(self)[index]

    def __iter__(self) -> Iterator[T]:
        """Итератор по элементам множества"""
        return super().__iter__()

    def __str__(self) -> str:
        """Строковое представление множества"""
        elements = ", ".join(str(x) for x in self)
        return f"TSet{{{elements}}}"

    def __repr__(self) -> str:
        """Представление для отладки"""
        return f"TSet({super().__repr__()})"


def demonstrate_int_set():
    """Демонстрация работы с целыми числами"""
    print("=== Демонстрация TSet с целыми числами ===")

    set_a = TSet([1, 2, 3, 4, 5])
    set_b = TSet([3, 4, 5, 6, 7])
    print(f"Множество A: {set_a}")
    print(f"Множество B: {set_b}")
    print(f"Размер A: {set_a.size()}")
    print(f"Пусто ли A: {set_a.is_empty()}")
    print(f"Содержит ли A число 3: {set_a.contains(3)}")

    union_set = set_a.union(set_b)
    diff_set = set_a.difference(set_b)
    inter_set = set_a.intersection(set_b)
    print(f"Объединение A и B: {union_set}")
    print(f"Разность A и B: {diff_set}")
    print(f"Пересечение A и B: {inter_set}")

    print("Элементы множества A:")
    for i in range(set_a.size()):
        element = set_a.get_element(i)
        print(f" Элемент {i}: {element}")

    print("\nДемонстрация добавления и удаления:")
    set_a.add(10)
    print(f"После добавления 10: {set_a}")
    set_a.remove(3)
    print(f"После удаления 3: {set_a}")

    set_a.clear()
    print(f"После очистки: {set_a}, пустое: {set_a.is_empty()}")
    print()


def demonstrate_frac_set():
    """Демонстрация работы с дробями"""
    print("=== Демонстрация TSet с дробями TFrac ===")

    frac1 = TFrac(1, 2)
    frac2 = TFrac(3, 4)
    frac3 = TFrac(2, 4)
    frac4 = TFrac(5, 6)
    print(f"Дробь 1: {frac1}")
    print(f"Дробь 2: {frac2}")
    print(f"Дробь 3: {frac3}")
    print(f"Дробь 4: {frac4}")
    print(f"Дробь 1 == Дробь 3: {frac1 == frac3}")

    set_x = TSet([frac1, frac2])
    set_y = TSet([frac3, frac4])
    print(f"Множество X: {set_x}")
    print(f"Множество Y: {set_y}")

    union_frac = set_x.union(set_y)
    diff_frac = set_x.difference(set_y)
    inter_frac = set_x.intersection(set_y)
    print(f"Объединение X и Y: {union_frac}")
    print(f"Разность X и Y: {diff_frac}")
    print(f"Пересечение X и Y: {inter_frac}")

    test_frac = TFrac(2, 4)
    print(f"Содержит ли X дробь 2/4: {set_x.contains(test_frac)}")
    print(f"Содержит ли Y дробь 1/2: {set_y.contains(TFrac(1, 2))}")

    print("Элементы множества X:")
    for i in range(set_x.size()):
        element = set_x.get_element(i)
        print(f" Элемент {i}: {element}")
    print()


def demonstrate_error_handling():
    """Демонстрация обработки ошибок"""
    print("=== Демонстрация обработки ошибок ===")

    set_empty = TSet()
    set_numbers = TSet([1, 2, 3])
    print("Пустое множество:", set_empty)
    print("Множество с числами:", set_numbers)

    print("\n1. Попытка удаления несуществующего элемента:")
    try:
        set_empty.remove(1)
        print("Удаление выполнено успешно")
    except KeyError as e:
        print(f"Ошибка при удалении: {e}")

    print("\n2. Попытка получения элемента по неверному индексу:")
    try:
        element = set_empty.get_element(0)
        print(f"Получен элемент: {element}")
    except IndexError as e:
        print(f"Ошибка при получении элемента: {e}")

    print("\n3. Попытка создания дроби с нулевым знаменателем:")
    try:
        invalid_frac = TFrac(1, 0)
        print(f"Дробь создана: {invalid_frac}")
    except ValueError as e:
        print(f"Ошибка создания дроби: {e}")

    print("\n4. Попытка объединения с неправильным типом:")
    try:
        result = set_numbers.union([4, 5, 6])
        print(f"Объединение выполнено: {result}")
    except TypeError as e:
        print(f"Ошибка при объединении: {e}")
    print()


def demonstrate_all_methods():
    """Демонстрация всех методов класса TSet"""
    print("=== Демонстрация всех методов TSet ===")

    print("1. Конструктор:")
    set1 = TSet([10, 20, 30])
    set2 = TSet()
    print(f" Множество 1: {set1}")
    print(f" Множество 2: {set2}")

    print("\n2. Метод add():")
    set2.add(100)
    set2.add(200)
    set2.add(100)
    print(f" После добавления: {set2}")

    print("\n3. Метод remove():")
    set1.remove(20)
    print(f" После удаления 20: {set1}")

    print("\n4. Метод is_empty():")
    print(f" set1 пустое: {set1.is_empty()}")
    print(f" set2 пустое: {set2.is_empty()}")

    print("\n5. Метод contains():")
    print(f" set1 содержит 10: {set1.contains(10)}")
    print(f" set1 содержит 50: {set1.contains(50)}")

    print("\n6. Метод union():")
    set3 = TSet([30, 40, 50])
    union_set = set1.union(set3)
    print(f" {set1} ∪ {set3} = {union_set}")

    print("\n7. Метод difference():")
    diff_set = set1.difference(set3)
    print(f" {set1} - {set3} = {diff_set}")

    print("\n8. Метод intersection():")
    inter_set = set1.intersection(set3)
    print(f" {set1} ∩ {set3} = {inter_set}")

    print("\n9. Метод size():")
    print(f" Размер set1: {set1.size()}")
    print(f" Размер union_set: {union_set.size()}")

    print("\n10. Метод get_element():")
    print(" Элементы set1 по индексам:")
    for i in range(set1.size()):
        print(f" Индекс {i}: {set1.get_element(i)}")

    print("\n11. Метод clear():")
    set1.clear()
    print(f" После очистки: {set1}, пустое: {set1.is_empty()}")


def main():
    """Основная функция демонстрации"""
    demonstrate_int_set()
    demonstrate_frac_set()
    demonstrate_error_handling()
    demonstrate_all_methods()
    print("\n=== Демонстрация завершена ===")


if __name__ == "__main__":
    main()
