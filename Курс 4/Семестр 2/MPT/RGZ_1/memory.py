from __future__ import annotations
from enum import Enum
from typing import Optional

from anumber import TANumber, ZERO_STRING, TPNumber


class MemoryState(Enum):
    OFF = "_Off"
    ON = "_On"


class TMemory:

    def __init__(self, number: TANumber):
        self._mem_state = MemoryState.OFF
        self._mem: TANumber = number.copy()


    def mem_store(self, n: TANumber):
        self._mem = n.copy()
        self._mem_state = MemoryState.ON

    def mem_restore(self):
        return self._mem.copy()

    def mem_add(self, n: TANumber):
        self._mem = self._mem.add(n)
        self._mem_state = MemoryState.ON

    def mem_clear(self):
        self._mem = TPNumber(ZERO_STRING)
        self._mem_state = MemoryState.OFF

    @property
    def mem_on(self):
        return self._mem_state.value

    @property
    def number(self):
        return self._mem.string
