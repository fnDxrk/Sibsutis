# editor.py

class Editor:
    
    def __init__(self):
        self.number = ""
        self.delim = "."
        self.zero = "0"
    
    @property
    def Number(self) :
        return self.number if self.number else self.zero
    
    def AddDigit(self, n: int) :
        if 0 <= n <= 15:
            digit = str(n) if n <= 9 else chr(ord('A') + n - 10)
            if self.number == self.zero:
                self.number = digit
            else:
                self.number += digit
        return self.Number
    
    def AddZero(self):
        if self.number and self.number != self.zero:
            self.number += self.zero
        return self.Number
    
    def AddDelim(self):
        if self.delim not in self.number:
            if not self.number:
                self.number = self.zero
            self.number += self.delim
        return self.Number
    
    def Bs(self) :
        if self.number:
            self.number = self.number[:-1]
        return self.Number
    
    def Clear(self) :
        self.number = ""
        return self.Number
    
    def Acc(self) -> int:
        if self.delim in self.number:
            return len(self.number.split(self.delim)[1])
        return 0
    
    def DoEdit(self, j: int) :
        if 0 <= j <= 15:
            return self.AddDigit(j)
        elif j == 16:
            return self.AddDelim()
        elif j == 17:
            return self.Bs()
        elif j == 18:
            return self.Clear()
        else:
            return self.Number


if __name__ == "__main__":
    print("=== Тест Editor ===\n")
    
    ed = Editor()
    
    print(f"Начало: '{ed.Number}'")
    print(f"AddDigit(1): '{ed.AddDigit(1)}'")
    print(f"AddDigit(5): '{ed.AddDigit(5)}'")
    print(f"AddDelim(): '{ed.AddDelim()}'")
    print(f"AddDigit(7): '{ed.AddDigit(7)}'")
    print(f"AddDigit(5): '{ed.AddDigit(5)}'")
    print(f"Текущее число: '{ed.Number}'")
    print(f"Точность: {ed.Acc()}")
    print(f"Bs(): '{ed.Bs()}'")
    print(f"Точность: {ed.Acc()}")
    print(f"Clear(): '{ed.Clear()}'")

