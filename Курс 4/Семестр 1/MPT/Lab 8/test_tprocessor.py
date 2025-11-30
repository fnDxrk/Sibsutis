import unittest
from typing import List
from tprocessor import TProc, TOprtn, TFunc


class TestTProc(unittest.TestCase):
    """Тесты для класса TProc"""

    def test_initialization(self):
        """Тест инициализации"""
        proc = TProc[int](0)
        self.assertEqual(proc.Lop_Res, 0)
        self.assertEqual(proc.Rop, 0)
        self.assertEqual(proc.Operation, TOprtn.None_)
        self.assertEqual(proc.ReadState(), TOprtn.None_)

    def test_reset_processor(self):
        """Тест сброса процессора"""
        proc = TProc[int](0)
        proc.WriteLeftOperand(10)
        proc.WriteRightOperand(5)
        proc.WriteState(TOprtn.Add)
        proc.ResetProcessor(0)
        self.assertEqual(proc.Lop_Res, 0)
        self.assertEqual(proc.Rop, 0)
        self.assertEqual(proc.Operation, TOprtn.None_)

    def test_reset_operation(self):
        """Тест сброса операции"""
        proc = TProc[int](0)
        proc.WriteState(TOprtn.Add)
        proc.ResetOperation()
        self.assertEqual(proc.Operation, TOprtn.None_)

    def test_execute_operation_add(self):
        """Тест выполнения операции сложения"""
        proc = TProc[int](0)
        proc.WriteLeftOperand(10)
        proc.WriteRightOperand(5)
        proc.WriteState(TOprtn.Add)
        proc.ExecuteOperation()
        self.assertEqual(proc.Lop_Res, 15)
        self.assertEqual(proc.Rop, 5)

    def test_execute_operation_sub(self):
        """Тест выполнения операции вычитания"""
        proc = TProc[int](0)
        proc.WriteLeftOperand(10)
        proc.WriteRightOperand(5)
        proc.WriteState(TOprtn.Sub)
        proc.ExecuteOperation()
        self.assertEqual(proc.Lop_Res, 5)

    def test_execute_operation_mul(self):
        """Тест выполнения операции умножения"""
        proc = TProc[int](0)
        proc.WriteLeftOperand(10)
        proc.WriteRightOperand(5)
        proc.WriteState(TOprtn.Mul)
        proc.ExecuteOperation()
        self.assertEqual(proc.Lop_Res, 50)

    def test_execute_operation_div(self):
        """Тест выполнения операции деления"""
        proc = TProc[float](0.0)
        proc.WriteLeftOperand(10.0)
        proc.WriteRightOperand(5.0)
        proc.WriteState(TOprtn.Dvd)
        proc.ExecuteOperation()
        self.assertEqual(proc.Lop_Res, 2.0)

    def test_execute_operation_none(self):
        """Тест выполнения операции None (ничего не должно происходить)"""
        proc = TProc[int](0)
        proc.WriteLeftOperand(10)
        proc.WriteRightOperand(5)
        proc.ExecuteOperation()
        self.assertEqual(proc.Lop_Res, 10)

    def test_execute_function_sqr(self):
        """Тест выполнения функции квадрата"""
        proc = TProc[int](0)
        proc.WriteRightOperand(5)
        proc.ExecuteFunction(TFunc.Sqr)
        self.assertEqual(proc.Rop, 25)

    def test_operand_read_write(self):
        """Тест чтения/записи операндов"""
        proc = TProc[int](0)
        proc.WriteLeftOperand(100)
        proc.WriteRightOperand(200)
        self.assertEqual(proc.ReadLeftOperand(), 100)
        self.assertEqual(proc.ReadRightOperand(), 200)

    def test_state_read_write(self):
        """Тест чтения/записи состояния"""
        proc = TProc[int](0)
        proc.WriteState(TOprtn.Mul)
        self.assertEqual(proc.ReadState(), TOprtn.Mul)

    def test_properties(self):
        """Тест свойств"""
        proc = TProc[int](0)
        proc.Lop_Res = 123
        proc.Rop = 456
        proc.Operation = TOprtn.Sub
        self.assertEqual(proc.Lop_Res, 123)
        self.assertEqual(proc.Rop, 456)
        self.assertEqual(proc.Operation, TOprtn.Sub)


class TestTProcWithList(unittest.TestCase):
    """Тесты для TProc с пользовательским типом List"""

    def test_list_operations(self):
        """Тест операций со списками"""
        proc = TProc[List[int]]([])
        proc.WriteLeftOperand([1, 2])
        proc.WriteRightOperand([3, 4])
        proc.WriteState(TOprtn.Add)
        proc.ExecuteOperation()
        self.assertEqual(proc.Lop_Res, [1, 2, 3, 4])

        with self.assertRaises(TypeError):
            proc.ExecuteFunction(TFunc.Sqr)


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
    suite.addTests(loader.loadTestsFromTestCase(TestTProc))
    suite.addTests(loader.loadTestsFromTestCase(TestTProcWithList))

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
