import pytest
from aeditor import PEditor, FEditor


class TestPEditor:
    def test_basic_edit(self):
        editor = PEditor()
        editor.edit(5)  # 5
        assert editor.string == "5"
        editor.edit(10)  # ± → -5
        assert editor.string == "-5"
        editor.edit(11)  # . → -5.
        assert editor.string == "-5."
        editor.edit(2)  # -5.2
        assert "5.2" in editor.string

    def test_backspace_clear(self):
        editor = PEditor()
        editor.edit(1);
        editor.edit(2)  # 12
        editor.edit(13)  # ⌫ → 1
        assert editor.string == "1"
        editor.edit(14)  # C → 0
        assert editor.string == "0"


class TestFEditor:
    def test_basic_input(self):
        editor = FEditor()
        editor.edit(1)
        editor.edit(11)
        editor.edit(2)
        assert editor.string == "1/2"
