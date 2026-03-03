from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar
import math

DECIMAL_SEPARATOR = "."
ZERO_STRING = "0"

TNumber = TypeVar("TNumber", bound="TANumber")


class TANumber(ABC):

    def __init__(self, value: str = ZERO_STRING) -> None:
        self._string = value

    @abstractmethod
    def is_zero(self):
        """"""

    @abstractmethod
    def copy(self: TNumber):
        """"""

    @abstractmethod
    def add(self: TNumber, other: TANumber):
        """"""

    @abstractmethod
    def sub(self: TNumber, other: TANumber):
        """"""

    @abstractmethod
    def mul(self: TNumber, other: TANumber):
        """"""

    @abstractmethod
    def div(self: TNumber, other: TANumber):
        """"""

    @abstractmethod
    def equals(self, other: TANumber):
        """"""

    @abstractmethod
    def sqr(self: TNumber):
        """"""

    @abstractmethod
    def inv(self: TNumber):
        """"""

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, value: str):
        self._string = value

    def __str__(self):
        return self.string

    def __repr__(self):
        return f"{self.__class__.__name__}({self.string!r})"


@dataclass
class TPNumber(TANumber):

    base: int = 10

    def __init__(self, value: str = ZERO_STRING, base: int = 10):
        super().__init__(value)
        if not (2 <= base <= 16):
            raise ValueError("Основание системы счисления должно быть в диапазоне 2..16")
        self.base = base

    def _to_float(self):
        return float(self.string.replace(DECIMAL_SEPARATOR, "."))

    def _from_float(self, x: float):
        if x == 0.0:
            return "0"
        s = f"{x:g}".replace(".", DECIMAL_SEPARATOR)
        return s


    def is_zero(self):
        try:
            return self._to_float() == 0.0
        except ValueError:
            return False

    def copy(self):
        return TPNumber(self.string, self.base)

    def add(self, other: TANumber):
        v = self._to_float() + TPNumber(str(other), self.base)._to_float()
        return TPNumber(self._from_float(v), self.base)

    def sub(self, other: TANumber):
        v = self._to_float() - TPNumber(str(other), self.base)._to_float()
        return TPNumber(self._from_float(v), self.base)

    def mul(self, other: TANumber):
        v = self._to_float() * TPNumber(str(other), self.base)._to_float()
        return TPNumber(self._from_float(v), self.base)

    def div(self, other: TANumber):
        other_val = TPNumber(str(other), self.base)._to_float()
        if other_val == 0.0:
            raise ZeroDivisionError("Деление на ноль")
        v = self._to_float() / other_val
        return TPNumber(self._from_float(v), self.base)

    def equals(self, other: TANumber):
        return self._to_float() == TPNumber(str(other), self.base)._to_float()

    def sqr(self):
        v = self._to_float() ** 2
        return TPNumber(self._from_float(v), self.base)

    def inv(self):
        if self.is_zero():
            raise ZeroDivisionError("Обратное к нулю не существует")
        v = 1.0 / self._to_float()
        return TPNumber(self._from_float(v), self.base)


@dataclass
class TFrac(TANumber):

    numerator: int = 0
    denominator: int = 1

    def __init__(self, value: str = ZERO_STRING):
        super().__init__(value)
        self._parse_from_string(value)


    def _parse_from_string(self, s: str):
        if "/" in s:
            num_s, den_s = s.split("/", 1)
            self.numerator = int(num_s)
            self.denominator = int(den_s)
        else:
            self.numerator = int(s)
            self.denominator = 1
        if self.denominator == 0:
            raise ZeroDivisionError("Знаменатель не может быть равен 0")
        self._normalize()

    def _normalize(self):
        if self.denominator < 0:
            self.denominator *= -1
            self.numerator *= -1
        g = math.gcd(self.numerator, self.denominator)
        if g != 0:
            self.numerator //= g
            self.denominator //= g
        self._update_string()

    def _update_string(self):
        if self.numerator == 0:
            self.string = "0"
        elif self.denominator == 1:
            self.string = str(self.numerator)
        else:
            self.string = f"{self.numerator}/{self.denominator}"


    def is_zero(self):
        return self.numerator == 0

    def copy(self):
        return TFrac(self.string)

    def add(self, other: TANumber):
        o = TFrac(str(other))
        n = self.numerator * o.denominator + self.denominator * o.numerator
        d = self.denominator * o.denominator
        return TFrac(f"{n}/{d}")

    def sub(self, other: TANumber):
        o = TFrac(str(other))
        n = self.numerator * o.denominator - self.denominator * o.numerator
        d = self.denominator * o.denominator
        return TFrac(f"{n}/{d}")

    def mul(self, other: TANumber):
        o = TFrac(str(other))
        n = self.numerator * o.numerator
        d = self.denominator * o.denominator
        return TFrac(f"{n}/{d}")

    def div(self, other: TANumber):
        o = TFrac(str(other))
        if o.numerator == 0:
            raise ZeroDivisionError("Деление на ноль")
        n = self.numerator * o.denominator
        d = self.denominator * o.numerator
        return TFrac(f"{n}/{d}")

    def equals(self, other: TANumber):
        o = TFrac(str(other))
        return self.numerator == o.numerator and self.denominator == o.denominator

    def sqr(self):
        n = self.numerator**2
        d = self.denominator**2
        return TFrac(f"{n}/{d}")

    def inv(self):
        if self.is_zero():
            raise ZeroDivisionError("Обратная дробь к нулю не существует")
        return TFrac(f"{self.denominator}/{self.numerator}")


@dataclass
class TComp(TANumber):

    re: TPNumber = field(default_factory=lambda: TPNumber(ZERO_STRING))
    im: TPNumber = field(default_factory=lambda: TPNumber(ZERO_STRING))

    COMPLEX_SEPARATOR = " i* "

    def __init__(self, value: str = ZERO_STRING):
        super().__init__(value)
        if value != ZERO_STRING:
            self._parse_from_string(value)


    def _parse_from_string(self, s: str):
        if self.COMPLEX_SEPARATOR in s:
            re_s, im_s = s.split(self.COMPLEX_SEPARATOR, 1)
            self.re = TPNumber(re_s)
            self.im = TPNumber(im_s)
        else:
            self.re = TPNumber(s)
            self.im = TPNumber(ZERO_STRING)
        self._update_string()

    def _update_string(self):
        self.string = f"{self.re.string}{self.COMPLEX_SEPARATOR}{self.im.string}"


    def is_zero(self):
        return self.re.is_zero() and self.im.is_zero()

    def copy(self):
        return TComp(self.string)

    def add(self, other: TANumber):
        o = TComp(str(other))
        re = self.re.add(o.re)
        im = self.im.add(o.im)
        return TComp(f"{re.string}{self.COMPLEX_SEPARATOR}{im.string}")

    def sub(self, other: TANumber):
        o = TComp(str(other))
        re = self.re.sub(o.re)
        im = self.im.sub(o.im)
        return TComp(f"{re.string}{self.COMPLEX_SEPARATOR}{im.string}")

    def mul(self, other: TANumber):
        o = TComp(str(other))
        a, b = self.re._to_float(), self.im._to_float()
        c, d = o.re._to_float(), o.im._to_float()
        re_val = a * c - b * d
        im_val = a * d + b * c
        re_new = TPNumber(str(re_val))
        im_new = TPNumber(str(im_val))
        return TComp(f"{re_new.string}{self.COMPLEX_SEPARATOR}{im_new.string}")

    def div(self, other: TANumber):
        o = TComp(str(other))
        a, b = self.re._to_float(), self.im._to_float()
        c, d = o.re._to_float(), o.im._to_float()
        denom = c * c + d * d
        if denom == 0.0:
            raise ZeroDivisionError("Деление на ноль")
        re_val = (a * c + b * d) / denom
        im_val = (b * c - a * d) / denom
        re_new = TPNumber(str(re_val))
        im_new = TPNumber(str(im_val))
        return TComp(f"{re_new.string}{self.COMPLEX_SEPARATOR}{im_new.string}")

    def equals(self, other: TANumber):
        o = TComp(str(other))
        return self.re.equals(o.re) and self.im.equals(o.im)

    def sqr(self):
        return self.mul(self)

    def inv(self):
        if self.is_zero():
            raise ZeroDivisionError("Обратное к нулю комплексное число не существует")
        one = TComp("1 i* 0")
        return one.div(self)
