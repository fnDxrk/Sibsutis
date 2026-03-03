import pytest
from memory import TMemory
from anumber import TPNumber


def test_memory_operations():
    num1 = TPNumber("123")
    mem = TMemory(num1)

    num2 = TPNumber("456")
    mem.mem_store(num2)
    assert mem.mem_on == "_On"

    restored = mem.mem_restore()
    assert restored.string == "456"

    mem.mem_clear()
    assert mem.mem_on == "_Off"
