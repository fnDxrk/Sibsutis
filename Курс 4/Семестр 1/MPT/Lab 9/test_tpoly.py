import unittest
from tpoly import TMember, TPoly


class TestTMember(unittest.TestCase):
    """Тесты для класса TMember"""

    def test_initialization(self):
        """Тест инициализации одночлена"""
        m1 = TMember(3, 2)
        self.assertEqual(m1.get_coeff(), 3)
        self.assertEqual(m1.get_degree(), 2)

        m2 = TMember(0, 5)
        self.assertEqual(m2.get_coeff(), 0)
        self.assertEqual(m2.get_degree(), 0)

    def test_setters(self):
        """Тест установки значений"""
        m = TMember(3, 2)
        m.set_coeff(5)
        m.set_degree(4)
        self.assertEqual(m.get_coeff(), 5)
        self.assertEqual(m.get_degree(), 4)

        m.set_coeff(0)
        self.assertEqual(m.get_degree(), 0)

    def test_equality(self):
        """Тест сравнения одночленов"""
        m1 = TMember(3, 2)
        m2 = TMember(3, 2)
        m3 = TMember(3, 3)
        self.assertTrue(m1 == m2)
        self.assertFalse(m1 == m3)

    def test_differentiate(self):
        """Тест дифференцирования одночлена"""
        m1 = TMember(4, 3)
        dm1 = m1.differentiate()
        self.assertEqual(dm1.get_coeff(), 12)
        self.assertEqual(dm1.get_degree(), 2)

        m2 = TMember(5, 0)
        dm2 = m2.differentiate()
        self.assertEqual(dm2.get_coeff(), 0)

    def test_evaluate(self):
        """Тест вычисления одночлена"""
        m = TMember(4, 3)
        self.assertEqual(m.evaluate(2), 32)

    def test_to_string(self):
        """Тест строкового представления"""
        self.assertEqual(TMember(3, 2).to_string(), "3x^2")
        self.assertEqual(TMember(-1, 3).to_string(), "-x^3")
        self.assertEqual(TMember(0, 5).to_string(), "0")
        self.assertEqual(TMember(5, 0).to_string(), "5")


class TestTPoly(unittest.TestCase):
    """Тесты для класса TPoly"""

    def test_initialization(self):
        """Тест инициализации полинома"""
        p1 = TPoly(2, 3)
        self.assertEqual(p1.get_degree(), 3)
        self.assertEqual(p1.get_coeff(3), 2)

        p2 = TPoly()
        self.assertEqual(p2.get_degree(), 0)

    def test_add(self):
        """Тест сложения полиномов"""
        p1 = TPoly(2, 3)
        p2 = TPoly(3, 1)
        p3 = p1.add(p2)
        self.assertEqual(p3.get_coeff(3), 2)
        self.assertEqual(p3.get_coeff(1), 3)

    def test_subtract(self):
        """Тест вычитания полиномов"""
        p1 = TPoly(5, 2)
        p2 = TPoly(3, 2)
        p3 = p1.subtract(p2)
        self.assertEqual(p3.get_coeff(2), 2)

    def test_multiply(self):
        """Тест умножения полиномов"""
        p1 = TPoly(1, 1)
        p2 = TPoly(1, 1)
        p3 = p1.multiply(p2)
        self.assertEqual(p3.get_degree(), 2)
        self.assertEqual(p3.get_coeff(2), 1)

    def test_negate(self):
        """Тест унарного минуса"""
        p1 = TPoly(2, 3)
        p2 = p1.negate()
        self.assertEqual(p2.get_coeff(3), -2)

    def test_equality(self):
        """Тест сравнения полиномов"""
        p1 = TPoly(3, 1)
        p2 = TPoly(3, 1)
        p3 = TPoly(2, 1)
        self.assertTrue(p1 == p2)
        self.assertFalse(p1 == p3)

    def test_differentiate(self):
        """Тест дифференцирования полинома"""
        p1 = TPoly(1, 3).add(TPoly(7, 1)).add(TPoly(5, 0))
        dp1 = p1.differentiate()
        self.assertEqual(dp1.get_coeff(2), 3)
        self.assertEqual(dp1.get_coeff(0), 7)

    def test_evaluate(self):
        """Тест вычисления полинома"""
        p1 = TPoly(1, 2).add(TPoly(3, 1))
        self.assertEqual(p1.evaluate(2), 10)

    def test_clear(self):
        """Тест очистки полинома"""
        p = TPoly(5, 3)
        p.clear()
        self.assertEqual(p.get_degree(), 0)

    def test_get_member(self):
        """Тест доступа к членам полинома"""
        p = TPoly(1, 3).add(TPoly(7, 1))
        coeff, degree = p.get_member(0)
        self.assertEqual(degree, 3)
        self.assertEqual(coeff, 1)

        with self.assertRaises(IndexError):
            p.get_member(10)


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
    suite.addTests(loader.loadTestsFromTestCase(TestTPoly))

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
