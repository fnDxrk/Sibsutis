import unittest
from tpnumber import TPNumber, TPNumberException


class TestTPNumber(unittest.TestCase):
    def test_constructor_numbers(self):
        # Тест конструктора с числами
        num = TPNumber(10.5, 10, 2)
        self.assertEqual(num.get_number(), 10.5)
        self.assertEqual(num.get_base_number(), 10)
        self.assertEqual(num.get_precision_number(), 2)

    def test_constructor_strings(self):
        # Тест конструктора со строками
        num = TPNumber("10.5", "10", "2")
        self.assertEqual(num.get_number(), 10.5)
        self.assertEqual(num.get_base_number(), 10)
        self.assertEqual(num.get_precision_number(), 2)

    def test_constructor_invalid_base(self):
        # Тест с недопустимым основанием
        with self.assertRaises(TPNumberException):
            TPNumber(10, 1, 2)
        with self.assertRaises(TPNumberException):
            TPNumber(10, 17, 2)

    def test_constructor_invalid_precision(self):
        # Тест с отрицательной точностью
        with self.assertRaises(TPNumberException):
            TPNumber(10, 10, -1)

    def test_copy(self):
        # Тест копирования
        num1 = TPNumber(10.5, 10, 2)
        num2 = num1.copy()
        self.assertEqual(num1.get_number(), num2.get_number())
        self.assertEqual(num1.get_base_number(), num2.get_base_number())
        self.assertEqual(num1.get_precision_number(), num2.get_precision_number())

    def test_add(self):
        # Тест сложения
        num1 = TPNumber(10.5, 10, 2)
        num2 = TPNumber(5.5, 10, 2)
        result = num1.add(num2)
        self.assertEqual(result.get_number(), 16.0)

    def test_multiply(self):
        # Тест умножения
        num1 = TPNumber(10.0, 10, 2)
        num2 = TPNumber(2.5, 10, 2)
        result = num1.multiply(num2)
        self.assertEqual(result.get_number(), 25.0)

    def test_subtract(self):
        # Тест вычитания
        num1 = TPNumber(10.0, 10, 2)
        num2 = TPNumber(3.5, 10, 2)
        result = num1.subtract(num2)
        self.assertEqual(result.get_number(), 6.5)

    def test_divide(self):
        # Тест деления
        num1 = TPNumber(10.0, 10, 2)
        num2 = TPNumber(4.0, 10, 2)
        result = num1.divide(num2)
        self.assertEqual(result.get_number(), 2.5)

    def test_divide_by_zero(self):
        # Тест деления на ноль
        num1 = TPNumber(10.0, 10, 2)
        num2 = TPNumber(0.0, 10, 2)
        with self.assertRaises(TPNumberException):
            num1.divide(num2)

    def test_invert(self):
        # Тест обратного числа
        num = TPNumber(4.0, 10, 2)
        result = num.invert()
        self.assertEqual(result.get_number(), 0.25)

    def test_invert_zero(self):
        # Тест обратного числа для нуля
        num = TPNumber(0.0, 10, 2)
        with self.assertRaises(TPNumberException):
            num.invert()

    def test_square(self):
        # Тест квадрата
        num = TPNumber(5.0, 10, 2)
        result = num.square()
        self.assertEqual(result.get_number(), 25.0)

    def test_get_number_string(self):
        # Тест получения числа как строки
        num = TPNumber(10.5, 10, 2)
        self.assertEqual(num.get_number_string(), "10.50")

    def test_get_number_string_binary(self):
        # Тест получения числа в двоичной системе
        num = TPNumber(5.5, 2, 4)
        self.assertEqual(num.get_number_string(), "101.1000")

    def test_get_number_string_hex(self):
        # Тест получения числа в шестнадцатеричной системе
        num = TPNumber(255.5, 16, 2)
        self.assertEqual(num.get_number_string(), "ff.80")

    def test_set_base_number(self):
        # Тест установки основания (число)
        num = TPNumber(10.0, 10, 2)
        num.set_base_number(16)
        self.assertEqual(num.get_base_number(), 16)

    def test_set_base_string(self):
        # Тест установки основания (строка)
        num = TPNumber(10.0, 10, 2)
        num.set_base_string("16")
        self.assertEqual(num.get_base_number(), 16)

    def test_set_precision_number(self):
        # Тест установки точности (число)
        num = TPNumber(10.0, 10, 2)
        num.set_precision_number(4)
        self.assertEqual(num.get_precision_number(), 4)

    def test_set_precision_string(self):
        # Тест установки точности (строка)
        num = TPNumber(10.0, 10, 2)
        num.set_precision_string("4")
        self.assertEqual(num.get_precision_number(), 4)

    def test_different_bases_error(self):
        # Тест ошибки при разных основаниях
        num1 = TPNumber(10.0, 10, 2)
        num2 = TPNumber(5.0, 16, 2)
        with self.assertRaises(TPNumberException):
            num1.add(num2)

    def test_different_precisions_error(self):
        # Тест ошибки при разных точностях
        num1 = TPNumber(10.0, 10, 2)
        num2 = TPNumber(5.0, 10, 4)
        with self.assertRaises(TPNumberException):
            num1.add(num2)


if __name__ == "__main__":
    from rich import print
    from rich.panel import Panel
    from rich.console import Console
    from unittest import TextTestRunner, TestResult

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

    runner = TextTestRunner(resultclass=RichTestResult, verbosity=0)
    result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestTPNumber))

    console.print(
        Panel.fit(
            f"[green]Passed: {result.testsRun - len(result.failures) - len(result.errors)}[/green]\n"
            f"[red]Failed: {len(result.failures)}[/red]\n"
            f"[magenta]Errors: {len(result.errors)}[/magenta]\n"
            f"[yellow]Total: {result.testsRun}[/yellow]",
            title="Test Results",
        )
    )
