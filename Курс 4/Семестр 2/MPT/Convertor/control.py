import math
from editor import Editor
from conver_10_p import Conver_10_P
from conver_p_10 import Conver_P_10


class Control:
    
    def __init__(self):
        self.ed = Editor()
        self.Pin = 10
        self.Pout = 16
        self.accuracy = 10
    
    def acc(self) -> int:
        digits_in = self.ed.Acc()
        if digits_in == 0:
            return self.accuracy
        
        ratio = math.log(self.Pin) / math.log(self.Pout)
        return int(round(digits_in * ratio + 0.5))
    
    def DoCmnd(self, j: int) -> str:
        if j == 19:
            try:
                if self.Pin == 10:
                    num = float(self.ed.Number)
                    result = Conver_10_P.Do(num, self.Pout, self.acc())
                elif self.Pout == 10:
                    result = str(Conver_P_10.dval(self.ed.Number, self.Pin))
                else:
                    temp = Conver_P_10.dval(self.ed.Number, self.Pin)
                    result = Conver_10_P.Do(temp, self.Pout, self.acc())
                
                return result
            except Exception as e:
                return f"Ошибка: {e}"
        else:
            return self.ed.DoEdit(j)


if __name__ == "__main__":
    print("=== Тест Control ===\n")
    
    ctl = Control()
    
    ctl.Pin = 10
    ctl.Pout = 16
    
    print(f"Ввод: {ctl.DoCmnd(1)}")
    print(f"Ввод: {ctl.DoCmnd(7)}")
    print(f"Ввод: {ctl.DoCmnd(16)}")
    print(f"Ввод: {ctl.DoCmnd(8)}")
    print(f"Ввод: {ctl.DoCmnd(7)}")
    print(f"Ввод: {ctl.DoCmnd(5)}")
    print(f"Число: '{ctl.ed.Number}' (основание {ctl.Pin})")
    print(f"Точность: {ctl.acc()}")
    print(f"Результат (основание {ctl.Pout}): {ctl.DoCmnd(19)}")
    print()
    
    ctl.ed.Clear()
    ctl.Pin = 16
    ctl.Pout = 10
    
    print(f"Ввод: {ctl.DoCmnd(10)}")
    print(f"Ввод: {ctl.DoCmnd(1)}")
    print(f"Ввод: {ctl.DoCmnd(16)}")
    print(f"Ввод: {ctl.DoCmnd(14)}")
    print(f"Число: '{ctl.ed.Number}' (основание {ctl.Pin})")
    print(f"Результат (основание {ctl.Pout}): {ctl.DoCmnd(19)}")

