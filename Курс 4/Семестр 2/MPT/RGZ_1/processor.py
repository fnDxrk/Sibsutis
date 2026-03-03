from __future__ import annotations
from enum import Enum, auto
from typing import Optional

from anumber import TANumber, TPNumber, TFrac, TComp, ZERO_STRING


class TOprtn(Enum):
    NONE = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DVD = auto()


class TFunc(Enum):
    REV = auto()
    SQR = auto()


class TProc:

    def __init__(self, left: TANumber, right: TANumber):
        self._lop_res: TANumber = left.copy()
        self._rop: TANumber = right.copy()
        self._operation: TOprtn = TOprtn.NONE
        self._error: str = ""


    @property
    def lop_res(self):
        return self._lop_res.copy()

    @lop_res.setter
    def lop_res(self, operand: TANumber):
        self._lop_res = operand.copy()

    @property
    def rop(self):
        return self._rop.copy()

    @rop.setter
    def rop(self, operand: TANumber):
        self._rop = operand.copy()

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, op: TOprtn):
        self._operation = op

    @property
    def error(self):
        return self._error

    def clear_error(self):
        self._error = ""

    def reset(self):
        self._lop_res = TPNumber(ZERO_STRING)
        self._rop = TPNumber(ZERO_STRING)
        self._operation = TOprtn.NONE
        self._error = ""

    def oprtn_clear(self):
        self._operation = TOprtn.NONE

    def oprtn_run(self):
        if self._operation == TOprtn.NONE:
            return
        try:
            if self._operation == TOprtn.ADD:
                self._lop_res = self._lop_res.add(self._rop)
            elif self._operation == TOprtn.SUB:
                self._lop_res = self._lop_res.sub(self._rop)
            elif self._operation == TOprtn.MUL:
                self._lop_res = self._lop_res.mul(self._rop)
            elif self._operation == TOprtn.DVD:
                self._lop_res = self._lop_res.div(self._rop)
        except Exception as ex:
            self._error = str(ex)

    def func_run(self, func: TFunc):
        try:
            if func == TFunc.REV:
                self._rop = self._rop.inv()
            elif func == TFunc.SQR:
                self._rop = self._rop.sqr()
        except Exception as ex:
            self._error = str(ex)
