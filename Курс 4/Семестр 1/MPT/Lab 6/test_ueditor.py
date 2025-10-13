import unittest
from ueditor import TEditor


class TestTEditor(unittest.TestCase):
    def setUp(self):
        self.editor = TEditor()

    def test_initial_state(self):
        self.assertEqual(self.editor.string, "0, i* 0,")
        self.assertTrue(self.editor.is_zero())

    def test_clear(self):
        self.editor.string = "123, i* 456,"
        result = self.editor.clear()
        self.assertEqual(result, "0, i* 0,")
        self.assertTrue(self.editor.is_zero())

    def test_backspace(self):
        self.editor.string = "123, i* 456,"
        result = self.editor.backspace()
        self.assertEqual(result, "123, i* 456")
        # Test backspace on zero
        self.editor.clear()
        result = self.editor.backspace()
        self.assertEqual(result, "0, i* 0,")

    def test_add_sign(self):
        result = self.editor.add_sign()
        self.assertEqual(result, "-0, i* 0,")
        result = self.editor.add_sign()  # Toggle back
        self.assertEqual(result, "0, i* 0,")

    def test_add_digit(self):
        result = self.editor.add_digit(5)
        self.assertEqual(result, "5, i* 0,")
        result = self.editor.add_digit(3)
        self.assertEqual(result, "53, i* 0,")

    def test_add_zero(self):
        result = self.editor.add_zero()
        self.assertEqual(result, "0, i* 0,")

    def test_edit_commands(self):
        # Test clear command
        result = self.editor.edit(0)
        self.assertEqual(result, "0, i* 0,")
        # Test add digit commands
        result = self.editor.edit(7)  # digit 3 (7-4=3)
        self.assertEqual(result, "3, i* 0,")
        # Test add sign command
        result = self.editor.edit(2)
        self.assertEqual(result, "-3, i* 0,")

    def test_add_decimal_separator(self):
        self.editor.string = "123 i* 456"
        result = self.editor.add_decimal_separator()
        self.assertEqual(result, "123, i* 456")
        result = self.editor.add_decimal_separator()  # Add to imaginary
        self.assertEqual(result, "123, i* 456,")

    def test_invalid_digit(self):
        with self.assertRaises(ValueError):
            self.editor.add_digit(10)

    def test_invalid_command(self):
        with self.assertRaises(ValueError):
            self.editor.edit(100)


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
    result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestTEditor))

    console.print(
        Panel.fit(
            f"[green]Passed: {result.testsRun - len(result.failures) - len(result.errors)}[/green]\n"
            f"[red]Failed: {len(result.failures)}[/red]\n"
            f"[magenta]Errors: {len(result.errors)}[/magenta]\n"
            f"[yellow]Total: {result.testsRun}[/yellow]",
            title="Test Results",
        )
    )
