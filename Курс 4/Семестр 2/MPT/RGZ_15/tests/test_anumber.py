import pytest
from anumber import TPNumber, TFrac, TComp

approx = pytest.approx


class TestTPNumber:
    def test_basic_arithmetic(self):
        a = TPNumber("5")
        b = TPNumber("3")
        assert a.add(b).string == "8"
        assert a.sub(b).string == "2"
        assert a.mul(b).string == "15"
        result = float(a.div(b).string)
        assert 1.6 < result < 1.7

    def test_negative(self):
        a = TPNumber("-5")
        result = a.add(TPNumber("3"))
        assert result.string == "-2"

    def test_functions(self):
        a = TPNumber("4")
        assert a.sqr().string == "16"
        inv_result = float(a.inv().string)
        assert 0.24 < inv_result < 0.26


class TestTFrac:
    def test_arithmetic(self):
        a = TFrac("1/2")
        b = TFrac("1/3")
        assert a.add(b).string == "5/6"


class TestTComp:
    def test_add(self):
        a = TComp("1 i* 2")
        b = TComp("3 i* 4")
        result = a.add(b)
        assert "4" in result.string

    def test_mul(self):
        a = TComp("0 i* 6")
        b = TComp("1 i* 1")
        result = a.mul(b)
        assert "i*" in result.string