import unittest
from matrix import Matrix


class TestMatrix(unittest.TestCase):
    def test_constructor_valid(self):
        """Тест конструктора с валидными данными"""
        data = [[1, 2], [3, 4]]
        matrix = Matrix(data)
        self.assertEqual(matrix.I, 2)
        self.assertEqual(matrix.J, 2)

    def test_constructor_invalid_empty(self):
        """Тест конструктора с пустой матрицей"""
        with self.assertRaises(ValueError):
            Matrix([])

    def test_constructor_invalid_empty_inner_array(self):
        """Тест конструктора с пустым внутренним массивом"""
        with self.assertRaises(ValueError):
            Matrix([[]])

    def test_constructor_invalid_irregular(self):
        """Тест конструктора с неправильной матрицей"""
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])

    def test_get_item_valid(self):
        """Тест получения элемента по индексу"""
        data = [[1, 2], [3, 4]]
        matrix = Matrix(data)
        self.assertEqual(matrix[0, 0], 1)
        self.assertEqual(matrix[1, 1], 4)

    def test_get_item_invalid(self):
        """Тест получения элемента с невалидным индексом"""
        data = [[1, 2], [3, 4]]
        matrix = Matrix(data)
        with self.assertRaises(IndexError):
            _ = matrix[2, 0]
        with self.assertRaises(IndexError):
            _ = matrix[0, 2]
        with self.assertRaises(IndexError):
            _ = matrix[-1, 0]
        with self.assertRaises(IndexError):
            _ = matrix[0, -1]

    def test_addition_valid(self):
        """Тест сложения матриц"""
        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[5, 6], [7, 8]])
        result = matrix1 + matrix2
        expected = Matrix([[6, 8], [10, 12]])
        self.assertEqual(result, expected)

    def test_addition_invalid_size(self):
        """Тест сложения матриц разного размера"""
        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            _ = matrix1 + matrix2

    def test_subtraction_valid(self):
        """Тест вычитания матриц"""
        matrix1 = Matrix([[5, 6], [7, 8]])
        matrix2 = Matrix([[1, 2], [3, 4]])
        result = matrix1 - matrix2
        expected = Matrix([[4, 4], [4, 4]])
        self.assertEqual(result, expected)

    def test_subtraction_invalid_size(self):
        """Тест вычитания матриц разного размера (покрытие исключения в sub)"""
        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            _ = matrix1 - matrix2

    def test_multiplication_valid(self):
        """Тест умножения матриц"""
        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[2, 0], [1, 2]])
        result = matrix1 * matrix2
        expected = Matrix([[4, 4], [10, 8]])
        self.assertEqual(result, expected)

    def test_multiplication_non_square(self):
        """Тест умножения неквадратных матриц"""
        matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
        matrix2 = Matrix([[1, 2], [3, 4], [5, 6]])
        result = matrix1 * matrix2
        expected = Matrix([[22, 28], [49, 64]])
        self.assertEqual(result, expected)

    def test_multiplication_invalid_size(self):
        """Тест умножения несогласованных матриц"""
        matrix1 = Matrix([[1, 2, 3]])
        matrix2 = Matrix([[1, 2]])
        with self.assertRaises(ValueError):
            _ = matrix1 * matrix2

    def test_equality_valid(self):
        """Тест проверки равенства матриц"""
        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[1, 2], [3, 4]])
        matrix3 = Matrix([[5, 6], [7, 8]])
        self.assertTrue(matrix1 == matrix2)
        self.assertFalse(matrix1 == matrix3)

    def test_equality_different_size(self):
        """Тест проверки равенства матриц разного размера"""
        matrix1 = Matrix([[1, 2]])
        matrix2 = Matrix([[1, 2, 3]])
        self.assertFalse(matrix1 == matrix2)

    def test_equality_with_different_values(self):
        """Тест проверки равенства матриц с разными значениями"""
        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[1, 2], [3, 5]])  # Одно значение отличается
        self.assertFalse(matrix1 == matrix2)

    def test_transpose_valid(self):
        """Тест транспонирования квадратной матрицы"""
        matrix = Matrix([[1, 2], [3, 4]])
        result = matrix.transp()
        expected = Matrix([[1, 3], [2, 4]])
        self.assertEqual(result, expected)

    def test_transpose_invalid(self):
        """Тест транспонирования неквадратной матрицы"""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            _ = matrix.transp()

    def test_min_element(self):
        """Тест поиска минимального элемента"""
        matrix = Matrix([[5, 2], [8, 1], [3, 7]])
        self.assertEqual(matrix.min(), 1)

    def test_min_element_single(self):
        """Тест поиска минимального элемента в матрице 1x1"""
        matrix = Matrix([[42]])
        self.assertEqual(matrix.min(), 42)

    def test_min_element_negative(self):
        """Тест поиска минимального отрицательного элемента"""
        matrix = Matrix([[5, -2], [8, 1], [3, -7]])
        self.assertEqual(matrix.min(), -7)

    def test_min_element_all_negative(self):
        """Тест поиска минимального элемента в матрице с отрицательными значениями"""
        matrix = Matrix([[-5, -2], [-8, -1], [-3, -7]])
        self.assertEqual(matrix.min(), -8)

    def test_min_element_empty_matrix(self):
        """Тест поиска минимального элемента в пустой матрице (покрытие исключения в min())"""
        # Создаем "пустую" матрицу для тестирования исключения
        matrix = Matrix.__new__(Matrix)
        matrix._data = []
        matrix._i = 0
        matrix._j = 0
        with self.assertRaises(ValueError):
            matrix.min()

    def test_to_string(self):
        """Тест преобразования в строку"""
        matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = "{{1,2,3},{4,5,6},{7,8,9}}"
        self.assertEqual(str(matrix), expected)

    def test_to_string_single_element(self):
        """Тест преобразования в строку матрицы 1x1"""
        matrix = Matrix([[42]])
        expected = "{{42}}"
        self.assertEqual(str(matrix), expected)

    def test_to_string_single_row(self):
        """Тест преобразования в строку матрицы с одной строкой"""
        matrix = Matrix([[1, 2, 3]])
        expected = "{{1,2,3}}"
        self.assertEqual(str(matrix), expected)

    def test_to_string_single_column(self):
        """Тест преобразования в строку матрицы с одним столбцом"""
        matrix = Matrix([[1], [2], [3]])
        expected = "{{1},{2},{3}}"
        self.assertEqual(str(matrix), expected)

    def test_repr_method(self):
        """Тест метода repr (покрытие метода repr)"""
        matrix = Matrix([[1, 2], [3, 4]])
        repr_str = repr(matrix)
        self.assertEqual(repr_str, "{{1,2},{3,4}}")
        self.assertEqual(
            repr_str, str(matrix)
        )  # repr должен возвращать то же, что и str

    def test_properties(self):
        """Тест свойств I и J"""
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(matrix.I, 2)
        self.assertEqual(matrix.J, 3)

    def test_properties_single_element(self):
        """Тест свойств I и J для матрицы 1x1"""
        matrix = Matrix([[42]])
        self.assertEqual(matrix.I, 1)
        self.assertEqual(matrix.J, 1)

    def test_properties_single_row(self):
        """Тест свойств I и J для матрицы с одной строкой"""
        matrix = Matrix([[1, 2, 3]])
        self.assertEqual(matrix.I, 1)
        self.assertEqual(matrix.J, 3)

    def test_properties_single_column(self):
        """Тест свойств I и J для матрицы с одним столбцом"""
        matrix = Matrix([[1], [2], [3]])
        self.assertEqual(matrix.I, 3)
        self.assertEqual(matrix.J, 1)


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
    result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestMatrix))

    console.print(
        Panel.fit(
            f"[green]Passed: {result.testsRun - len(result.failures) - len(result.errors)}[/green]\n"
            f"[red]Failed: {len(result.failures)}[/red]\n"
            f"[magenta]Errors: {len(result.errors)}[/magenta]\n"
            f"[yellow]Total: {result.testsRun}[/yellow]",
            title="Test Results",
        )
    )

