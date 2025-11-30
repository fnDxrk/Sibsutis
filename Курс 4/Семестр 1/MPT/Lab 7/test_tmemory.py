import unittest
from typing import List
from tmemory import TMemory


class TestTMemory(unittest.TestCase):
    """Тесты для класса TMemory"""
    
    def test_initialization(self):
        """Тест инициализации"""
        memory = TMemory[int](0)
        self.assertEqual(memory.FNumber, 0)
        self.assertEqual(memory.FState, "Выключена")
        self.assertEqual(memory.ReadMemoryState(), "Выключена")
        self.assertEqual(memory.ReadNumber(), 0)
    
    def test_store(self):
        """Тест операции Записать"""
        memory = TMemory[int](0)
        memory.Store(100)
        self.assertEqual(memory.FNumber, 100)
        self.assertEqual(memory.FState, "Включена")
        self.assertEqual(memory.ReadNumber(), 100)
    
    def test_get(self):
        """Тест операции Взять"""
        memory = TMemory[int](0)
        memory.Store(50)
        value = memory.Get()
        self.assertEqual(value, 50)
        self.assertEqual(memory.FState, "Включена")
    
    def test_add(self):
        """Тест операции Добавить"""
        memory = TMemory[int](10)
        memory.Store(20)  # Сначала записываем 20
        memory.Add(5)  # Добавляем 5 к 20 = 25
        self.assertEqual(memory.FNumber, 25)
        self.assertEqual(memory.FState, "Включена")
    
    def test_clear(self):
        """Тест операции Очистить"""
        memory = TMemory[int](0)
        memory.Store(999)
        memory.Clear(0)
        self.assertEqual(memory.FNumber, 0)
        self.assertEqual(memory.FState, "Выключена")
    
    def test_read_operations(self):
        """Тест операций чтения"""
        memory = TMemory[str]("default")
        memory.Store("test_value")
        state = memory.ReadMemoryState()
        number = memory.ReadNumber()
        self.assertEqual(state, "Включена")
        self.assertEqual(number, "test_value")
    
    def test_with_custom_type(self):
        """Тест с пользовательским типом (списком)"""
        memory = TMemory[List[int]]([])
        memory.Store([1, 2, 3])
        self.assertEqual(memory.Get(), [1, 2, 3])
        # Для списков операция Add будет конкатенацией
        memory.Add([4, 5])
        self.assertEqual(memory.Get(), [1, 2, 3, 4, 5])
    
    def test_add_unsupported_type(self):
        """Тест операции Добавить с неподдерживающим сложение типом"""
        memory = TMemory[dict]({})
        memory.Store({"a": 1})
        with self.assertRaises(TypeError):
            memory.Add({"b": 2})  # Словари не поддерживают операцию +


# Дополнительный пример с пользовательским классом
class Fraction:
    """Пример пользовательского класса для демонстрации"""
    
    def __init__(self, numerator: int = 0, denominator: int = 1):
        self.numerator = numerator
        self.denominator = denominator
    
    def __add__(self, other: 'Fraction') -> 'Fraction':
        # Простая реализация сложения дробей
        if self.denominator == other.denominator:
            return Fraction(self.numerator + other.numerator, self.denominator)
        return Fraction(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator
        )
    
    def __eq__(self, other: 'Fraction') -> bool:
        return (self.numerator == other.numerator and
                self.denominator == other.denominator)
    
    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


class TestTMemoryWithFraction(unittest.TestCase):
    """Тесты для TMemory с пользовательским типом Fraction"""
    
    def test_fraction_memory(self):
        """Тест памяти с дробями"""
        default_frac = Fraction(0, 1)
        memory = TMemory[Fraction](default_frac)
        
        # Записываем дробь
        frac1 = Fraction(1, 2)
        memory.Store(frac1)
        self.assertEqual(memory.Get().numerator, 1)
        self.assertEqual(memory.Get().denominator, 2)
        
        # Добавляем дробь
        frac2 = Fraction(1, 3)
        memory.Add(frac2)
        result = memory.Get()
        # 1/2 + 1/3 = 5/6
        self.assertEqual(result.numerator, 5)
        self.assertEqual(result.denominator, 6)
        
        # Очищаем
        memory.Clear(Fraction(0, 1))
        self.assertEqual(memory.ReadNumber().numerator, 0)  # Используем ReadNumber вместо Get
        self.assertEqual(memory.FState, "Выключена")  # Теперь состояние останется "Выключена"


if __name__ == "__main__":
    from rich import print
    from rich.panel import Panel
    from rich.console import Console
    from unittest import TextTestRunner, TestResult, TestSuite

    console = Console()

    class RichTestResult(TestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)

        def startTest(self, test):
            super().startTest(test)
            console.print(f"[cyan]Running:[/cyan] {test._testMethodName}")

        def addSuccess(self, test):
            super().addSuccess(test)
            console.print(f"[green]✓ PASS:[/green] {test._testMethodName}")

        def addFailure(self, test, err):
            super().addFailure(test, err)
            console.print(f"[red]✗ FAIL:[/red] {test._testMethodName}")

        def addError(self, test, err):
            super().addError(test, err)
            console.print(f"[magenta]💥 ERROR:[/magenta] {test._testMethodName}")

    loader = unittest.defaultTestLoader
    suite = TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestTMemoryWithFraction))

    runner = TextTestRunner(resultclass=RichTestResult, verbosity=0)
    result = runner.run(suite)

    console.print(
        Panel.fit(
            f"[green]Passed: {result.testsRun - len(result.failures) - len(result.errors)}[/green]\n"
            f"[red]Failed: {len(result.failures)}[/red]\n"
            f"[magenta]Errors: {len(result.errors)}[/magenta]\n"
            f"[yellow]Total: {result.testsRun}[/yellow]",
            title="Test Results",
        )
    )

