import unittest
from ProgramLengthSimulator import ProgramLengthSimulator
import math


class TestProgramLengthSimulator(unittest.TestCase):
    """Тесты для класса ProgramLengthSimulator"""

    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.simulator = ProgramLengthSimulator()

    def test_initialization(self):
        """Тест инициализации симулятора"""
        self.assertIsNotNone(self.simulator.results)
        self.assertIsNotNone(self.simulator.python_keywords)
        self.assertIsNotNone(self.simulator.builtin_names)
        self.assertIsInstance(self.simulator.results, dict)

    def test_simulate_program_writing(self):
        """Тест симуляции написания программы"""
        eta = 16
        lengths = self.simulator.simulate_program_writing(eta, num_trials=100)

        self.assertEqual(len(lengths), 100)
        self.assertIn(eta, self.simulator.results)
        self.assertIn("experimental", self.simulator.results[eta])
        self.assertIn("theoretical", self.simulator.results[eta])

    def test_simulate_program_writing_values(self):
        """Тест корректности значений симуляции"""
        eta = 32
        self.simulator.simulate_program_writing(eta, num_trials=1000)

        exp = self.simulator.results[eta]["experimental"]

        self.assertGreater(exp["mean_length"], eta)
        self.assertGreater(exp["variance"], 0)
        self.assertGreater(exp["std_dev"], 0)
        self.assertGreaterEqual(exp["relative_error"], 0)

    def test_calculate_theoretical_values(self):
        """Тест вычисления теоретических значений"""
        eta = 64
        theory = self.simulator.calculate_theoretical_values(eta)

        self.assertIn("mean_length", theory)
        self.assertIn("variance", theory)
        self.assertIn("std_dev", theory)
        self.assertIn("relative_error", theory)
        self.assertIn("alternative_length", theory)

        self.assertGreater(theory["mean_length"], 0)
        self.assertGreater(theory["variance"], 0)
        self.assertAlmostEqual(
            theory["std_dev"], math.sqrt(theory["variance"]), places=5
        )

    def test_theoretical_formulas(self):
        """Тест корректности теоретических формул"""
        eta = 128
        theory = self.simulator.calculate_theoretical_values(eta)

        expected_mean = 0.9 * eta * math.log2(eta)
        expected_variance = (math.pi**2 * eta**2) / 6
        expected_alt_length = eta * math.log2(eta)

        self.assertAlmostEqual(theory["mean_length"], expected_mean, places=5)
        self.assertAlmostEqual(theory["variance"], expected_variance, places=5)
        self.assertAlmostEqual(
            theory["alternative_length"], expected_alt_length, places=5
        )

    def test_analyze_simple_program(self):
        """Тест анализа простой программы"""
        program = """
x = 5
y = 10
z = x + y
print(z)
"""
        analysis = self.simulator.analyze_program_text(program)

        self.assertIn("eta", analysis)
        self.assertIn("actual_length", analysis)
        self.assertIn("predicted_length_1", analysis)
        self.assertIn("operators_count", analysis)
        self.assertIn("operands_count", analysis)

        self.assertGreater(analysis["eta"], 0)
        self.assertGreater(analysis["actual_length"], 0)

    def test_analyze_program_with_keywords(self):
        """Тест анализа программы с ключевыми словами"""
        program = """
if x > 0:
    for i in range(10):
        print(i)
"""
        analysis = self.simulator.analyze_program_text(program)

        self.assertGreater(analysis["operators_count"], 0)
        self.assertGreater(analysis["operands_count"], 0)

    def test_analyze_eta2_star(self):
        """Тест определения уникальных операндов"""
        program = """
x = 5
y = 10
z = x + y
result = x * y
"""
        operands = self.simulator.analyze_eta2_star(program)

        self.assertIsInstance(operands, set)
        self.assertGreater(len(operands), 0)
        self.assertIn("x", operands)
        self.assertIn("y", operands)
        self.assertIn("z", operands)

    def test_analyze_eta2_star_no_keywords(self):
        """Тест что ключевые слова не включаются в операнды"""
        program = """
if True:
    for i in range(10):
        pass
"""
        operands = self.simulator.analyze_eta2_star(program)

        self.assertNotIn("if", operands)
        self.assertNotIn("for", operands)
        self.assertNotIn("in", operands)
        self.assertNotIn("True", operands)

    def test_error_calculations(self):
        """Тест вычисления ошибок прогноза"""
        program = "x = 1 + 2"
        analysis = self.simulator.analyze_program_text(program)

        self.assertIn("error_1", analysis)
        self.assertIn("error_2", analysis)
        self.assertGreaterEqual(analysis["error_1"], 0)
        self.assertGreaterEqual(analysis["error_2"], 0)

    def test_empty_program(self):
        """Тест анализа пустой программы"""
        program = ""
        analysis = self.simulator.analyze_program_text(program)

        self.assertEqual(analysis["eta"], 0)
        self.assertEqual(analysis["actual_length"], 0)

    def test_multiple_eta_values(self):
        """Тест симуляции для нескольких значений eta"""
        eta_values = [16, 32, 64]

        for eta in eta_values:
            self.simulator.simulate_program_writing(eta, num_trials=100)
            self.assertIn(eta, self.simulator.results)

    def test_convergence_with_trials(self):
        """Тест сходимости результатов с увеличением числа испытаний"""
        eta = 32

        self.simulator.simulate_program_writing(eta, num_trials=100)
        result_100 = self.simulator.results[eta]["experimental"]["mean_length"]

        self.simulator.simulate_program_writing(eta, num_trials=10000)
        result_10000 = self.simulator.results[eta]["experimental"]["mean_length"]

        theory = self.simulator.results[eta]["theoretical"]["mean_length"]

        error_100 = abs(result_100 - theory) / theory
        error_10000 = abs(result_10000 - theory) / theory

        self.assertLessEqual(error_10000, error_100 * 2)


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
    suite.addTests(loader.loadTestsFromTestCase(TestProgramLengthSimulator))

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

