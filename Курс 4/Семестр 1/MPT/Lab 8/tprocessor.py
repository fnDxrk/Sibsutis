from typing import TypeVar, Generic, Optional, Union
from enum import Enum, auto

T = TypeVar("T")


class TOprtn(Enum):
    """Тип операций"""

    None_ = auto()
    Add = auto()
    Sub = auto()
    Mul = auto()
    Dvd = auto()


class TFunc(Enum):
    """Тип функций"""

    Rev = auto()
    Sqr = auto()


class TProc(Generic[T]):
    """Параметризованный класс Процессор для выполнения операций над значениями типа T"""

    def __init__(self, default_value: T):
        """
        Конструктор
        Начальные значения:
        - Lop_Res: объект типа T со значением по умолчанию
        - Rop: объект типа T со значением по умолчанию
        - Operation: None_ (операция не установлена)
        """
        self._Lop_Res = default_value
        self._Rop = default_value
        self._Operation = TOprtn.None_

    def ResetProcessor(self, default_value: T) -> None:
        """
        СбросПроцессора
        Процесс: Сбрасывает все поля к значениям по умолчанию
        """
        self._Lop_Res = default_value
        self._Rop = default_value
        self._Operation = TOprtn.None_

    def ResetOperation(self) -> None:
        """
        СбросОперации
        Процесс: Сбрасывает только операцию
        """
        self._Operation = TOprtn.None_

    def ExecuteOperation(self) -> None:
        """
        ВыполнитьОперацию
        Процесс: Выполняет текущую операцию над Lop_Res и Rop, результат сохраняет в Lop_Res
        """
        if self._Operation == TOprtn.None_:
            return

        try:
            if self._Operation == TOprtn.Add:
                self._Lop_Res = self._Lop_Res + self._Rop
            elif self._Operation == TOprtn.Sub:
                self._Lop_Res = self._Lop_Res - self._Rop
            elif self._Operation == TOprtn.Mul:
                self._Lop_Res = self._Lop_Res * self._Rop
            elif self._Operation == TOprtn.Dvd:
                self._Lop_Res = self._Lop_Res / self._Rop
        except ZeroDivisionError:
            raise ZeroDivisionError("Деление на ноль")
        except TypeError as e:
            raise TypeError(
                f"Операция не поддерживается для типа {type(self._Lop_Res).__name__}"
            ) from e

    def ExecuteFunction(self, func: TFunc) -> None:
        """
        ВычислитьФункцию
        Вход: func - тип функции
        Процесс: Выполняет функцию над Rop, результат сохраняет в Rop
        """
        try:
            if func == TFunc.Rev:
                self._Rop = 1 / self._Rop
            elif func == TFunc.Sqr:
                self._Rop = self._Rop * self._Rop
        except ZeroDivisionError:
            raise ZeroDivisionError("Обращение нуля")
        except TypeError as e:
            raise TypeError(
                f"Функция не поддерживается для типа {type(self._Rop).__name__}"
            ) from e

    def ReadLeftOperand(self) -> T:
        """
        ЧитатьЛевыйОперанд
        Выход: копия левого операнда
        """
        return self._Lop_Res

    def WriteLeftOperand(self, operand: T) -> None:
        """
        ЗаписатьЛевыйОперанд
        Вход: operand - объект типа T
        Процесс: Записывает копию operand в Lop_Res
        """
        self._Lop_Res = operand

    def ReadRightOperand(self) -> T:
        """
        ЧитатьПравыйОперанд
        Выход: копия правого операнда
        """
        return self._Rop

    def WriteRightOperand(self, operand: T) -> None:
        """
        ЗаписатьПравыйОперанд
        Вход: operand - объект типа T
        Процесс: Записывает копию operand в Rop
        """
        self._Rop = operand

    def ReadState(self) -> TOprtn:
        """
        ЧитатьСостояние
        Выход: текущая операция
        """
        return self._Operation

    def WriteState(self, oprtn: TOprtn) -> None:
        """
        ЗаписатьСостояние
        Вход: oprtn - тип операции
        Процесс: Устанавливает новую операцию
        """
        self._Operation = oprtn

    @property
    def Lop_Res(self) -> T:
        """Свойство для чтения/записи левого операнда-результата"""
        return self._Lop_Res

    @Lop_Res.setter
    def Lop_Res(self, value: T) -> None:
        self._Lop_Res = value

    @property
    def Rop(self) -> T:
        """Свойство для чтения/записи правого операнда"""
        return self._Rop

    @Rop.setter
    def Rop(self, value: T) -> None:
        self._Rop = value

    @property
    def Operation(self) -> TOprtn:
        """Свойство для чтения/записи операции"""
        return self._Operation

    @Operation.setter
    def Operation(self, value: TOprtn) -> None:
        self._Operation = value

    def __str__(self) -> str:
        """Строковое представление процессора"""
        op_map = {
            TOprtn.None_: "None",
            TOprtn.Add: "+",
            TOprtn.Sub: "-",
            TOprtn.Mul: "*",
            TOprtn.Dvd: "/",
        }
        return f"Процессор[Lop_Res: {self._Lop_Res}, Rop: {self._Rop}, Operation: {op_map[self._Operation]}]"


if __name__ == "__main__":
    proc = TProc[int](0)
    print(f"Создан процессор: {proc}")

    proc.WriteLeftOperand(10)
    proc.WriteRightOperand(5)
    print(f"После установки операндов: {proc}")

    proc.WriteState(TOprtn.Add)
    print(f"После установки операции: {proc}")

    proc.ExecuteOperation()
    print(f"После выполнения операции: {proc}")

    proc.WriteRightOperand(4)
    proc.ExecuteFunction(TFunc.Sqr)
    print(f"После вычисления квадрата: {proc}")

    proc.ResetProcessor(0)
    print(f"После сброса: {proc}")
