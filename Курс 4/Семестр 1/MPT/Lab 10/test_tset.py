import unittest
from tset import TSet, TFrac


class TestTFrac(unittest.TestCase):
    """Тесты для класса TFrac"""

    def test_initialization(self):
        """Тест инициализации дроби"""
        frac = TFrac(3, 4)
        self.assertEqual(frac.numerator, 3)
        self.assertEqual(frac.denominator, 4)

    def test_normalization(self):
        """Тест нормализации дроби"""
        frac = TFrac(2, 4)
        self.assertEqual(frac.numerator, 1)
        self.assertEqual(frac.denominator, 2)

        frac2 = TFrac(6, 9)
        self.assertEqual(frac2.numerator, 2)
        self.assertEqual(frac2.denominator, 3)

    def test_zero_denominator(self):
        """Тест создания дроби с нулевым знаменателем"""
        with self.assertRaises(ValueError):
            TFrac(1, 0)

    def test_equality(self):
        """Тест сравнения дробей"""
        frac1 = TFrac(1, 2)
        frac2 = TFrac(2, 4)
        frac3 = TFrac(3, 4)
        self.assertEqual(frac1, frac2)
        self.assertNotEqual(frac1, frac3)

    def test_hash(self):
        """Тест хеширования дробей"""
        frac1 = TFrac(1, 2)
        frac2 = TFrac(2, 4)
        self.assertEqual(hash(frac1), hash(frac2))

    def test_string_representation(self):
        """Тест строкового представления"""
        frac = TFrac(3, 4)
        self.assertEqual(str(frac), "3/4")


class TestTSet(unittest.TestCase):
    """Тесты для класса TSet"""

    def test_initialization(self):
        """Тест инициализации множества"""
        set1 = TSet([1, 2, 3])
        self.assertEqual(set1.size(), 3)

        set2 = TSet()
        self.assertEqual(set2.size(), 0)

    def test_add(self):
        """Тест добавления элемента"""
        set1 = TSet()
        set1.add(10)
        set1.add(20)
        set1.add(10)
        self.assertEqual(set1.size(), 2)

    def test_remove(self):
        """Тест удаления элемента"""
        set1 = TSet([1, 2, 3])
        set1.remove(2)
        self.assertEqual(set1.size(), 2)
        self.assertFalse(set1.contains(2))

        with self.assertRaises(KeyError):
            set1.remove(10)

    def test_is_empty(self):
        """Тест проверки пустоты"""
        set1 = TSet()
        self.assertTrue(set1.is_empty())

        set1.add(1)
        self.assertFalse(set1.is_empty())

    def test_contains(self):
        """Тест проверки принадлежности"""
        set1 = TSet([1, 2, 3])
        self.assertTrue(set1.contains(2))
        self.assertFalse(set1.contains(10))

    def test_union(self):
        """Тест объединения множеств"""
        set1 = TSet([1, 2, 3])
        set2 = TSet([3, 4, 5])
        union_set = set1.union(set2)
        self.assertEqual(union_set.size(), 5)
        self.assertTrue(union_set.contains(1))
        self.assertTrue(union_set.contains(5))

    def test_difference(self):
        """Тест разности множеств"""
        set1 = TSet([1, 2, 3, 4])
        set2 = TSet([3, 4, 5])
        diff_set = set1.difference(set2)
        self.assertEqual(diff_set.size(), 2)
        self.assertTrue(diff_set.contains(1))
        self.assertFalse(diff_set.contains(3))

    def test_intersection(self):
        """Тест пересечения множеств"""
        set1 = TSet([1, 2, 3, 4])
        set2 = TSet([3, 4, 5])
        inter_set = set1.intersection(set2)
        self.assertEqual(inter_set.size(), 2)
        self.assertTrue(inter_set.contains(3))
        self.assertFalse(inter_set.contains(1))

    def test_get_element(self):
        """Тест получения элемента по индексу"""
        set1 = TSet([10, 20, 30])
        element = set1.get_element(0)
        self.assertIn(element, [10, 20, 30])

        with self.assertRaises(IndexError):
            set1.get_element(10)

    def test_clear(self):
        """Тест очистки множества"""
        set1 = TSet([1, 2, 3])
        set1.clear()
        self.assertTrue(set1.is_empty())
        self.assertEqual(set1.size(), 0)

    def test_with_fractions(self):
        """Тест множества с дробями"""
        frac1 = TFrac(1, 2)
        frac2 = TFrac(3, 4)
        frac3 = TFrac(2, 4)

        set1 = TSet([frac1, frac2])
        self.assertEqual(set1.size(), 2)
        self.assertTrue(set1.contains(frac3))


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
    suite.addTests(loader.loadTestsFromTestCase(TestTFrac))
    suite.addTests(loader.loadTestsFromTestCase(TestTSet))

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
