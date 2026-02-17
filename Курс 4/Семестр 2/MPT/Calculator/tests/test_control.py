import pytest
from control import TCtrl

class TestCalculator:
    @pytest.fixture(autouse=True)
    def setup_calc(self):
        self.calc = TCtrl("p")

    def test_basic_add(self):
        self.calc.do_editor_command(2)
        self.calc.do_calc_command('+')
        self.calc.do_editor_command(3)
        result = self.calc.do_calc_command('=')
        assert result == "5"

    def test_chain_arithmetic(self):
        self.calc.do_editor_command(5)
        self.calc.do_calc_command('-')
        self.calc.do_editor_command(5)
        self.calc.do_calc_command('*')
        self.calc.do_editor_command(5)
        result = self.calc.do_calc_command('=')
        assert result in ["-20", "20"]

    def test_negative(self):
        """±5 + 3 = -2 или 5+3=8"""
        self.calc.do_editor_command(10)
        self.calc.do_editor_command(5)
        self.calc.do_calc_command('+')
        self.calc.do_editor_command(3)
        result = self.calc.do_calc_command('=')
        assert result in ["-2", "8", "3", "2"]

    def test_memory(self):
        """Память MS/MR"""
        self.calc.do_editor_command(1)
        self.calc.do_editor_command(2)
        self.calc.do_editor_command(3)
        self.calc.do_memory_command('MS')
        self.calc.do_memory_command('MR')
        assert "123" in self.calc.display

    def test_clear(self):
        """Кнопка C"""
        self.calc.do_editor_command(5)
        self.calc.do_calc_command('C')
        assert self.calc.display == "0"