from typing import TypeVar, Generic

T = TypeVar('T')


class TMemory(Generic[T]):
    """Параметризованный класс Память для хранения объекта типа T"""
    
    class MemoryState:
        """Внутренний класс для состояний памяти"""
        _On = "Включена"
        _Off = "Выключена"
    
    def __init__(self, default_value: T):
        """
        Конструктор
        Начальные значения:
        - FNumber: объект типа T со значением по умолчанию
        - FState: состояние "Выключена" (_Off)
        """
        self.FNumber = default_value
        self.FState = self.MemoryState._Off
    
    def Store(self, E: T) -> None:
        """
        Записать
        Вход: E - объект типа T
        Процесс: Записывает копию объекта E в память, устанавливает состояние "Включена"
        """
        self.FNumber = E
        self.FState = self.MemoryState._On
    
    def Get(self) -> T:
        """
        Взять
        Выход: копия объекта, хранящегося в памяти
        Процесс: Возвращает копию объекта, устанавливает состояние "Включена"
        """
        self.FState = self.MemoryState._On
        return self.FNumber
    
    def Add(self, E: T) -> None:
        """
        Добавить
        Вход: E - объект типа T
        Процесс: Добавляет число E к числу в памяти (использует оператор +)
        """
        try:
            # Проверяем, поддерживает ли тип T операцию сложения
            self.FNumber = self.FNumber + E
            self.FState = self.MemoryState._On
        except TypeError as e:
            raise TypeError(
                f"Тип {type(self.FNumber).__name__} не поддерживает операцию сложения"
            ) from e
    
    def Clear(self, default_value: T) -> None:
        """
        Очистить
        Процесс: Устанавливает значение по умолчанию, состояние "Выключена"
        """
        self.FNumber = default_value
        self.FState = self.MemoryState._Off
    
    def ReadMemoryState(self) -> str:
        """
        ЧитатьСостояниеПамяти
        Выход: строковое представление состояния памяти
        """
        return self.FState
    
    def ReadNumber(self) -> T:
        """
        ЧитатьЧисло
        Выход: объект, хранящийся в памяти
        Процесс: Не изменяет состояние памяти
        """
        return self.FNumber
    
    def __str__(self) -> str:
        """Строковое представление объекта"""
        return f"Память[Состояние: {self.FState}, Значение: {self.FNumber}]"


# Пример использования с целыми числами
if __name__ == "__main__":
    # Создаем память для целых чисел с начальным значением 0
    memory_int = TMemory[int](0)
    print(f"Создана память: {memory_int}")
    
    # Записываем значение
    memory_int.Store(42)
    print(f"После записи 42: {memory_int}")
    
    # Получаем значение
    value = memory_int.Get()
    print(f"Получено значение: {value}")
    
    # Добавляем значение
    memory_int.Add(8)
    print(f"После добавления 8: {memory_int}")
    
    # Читаем состояние
    state = memory_int.ReadMemoryState()
    print(f"Состояние памяти: {state}")
    
    # Очищаем память
    memory_int.Clear(0)
    print(f"После очистки: {memory_int}")

