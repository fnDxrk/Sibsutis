from cx_Freeze import setup, Executable
import sys

base = "gui" if sys.platform == "win32" else None

build_options = {
    "packages": ["PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets"],
    "excludes": [],
    "include_files": ["help.html"]
}

executables = [
    Executable(
        "ui_qt.py",
        base=base,
        target_name="Калькулятор.exe",
        icon="calculator.ico"
    )
]

setup(
    name="UniversalCalculator",
    version="1.0.0",
    description="Универсальный калькулятор СибГУТИ",
    options={"build_exe": build_options},
    executables=executables
)
