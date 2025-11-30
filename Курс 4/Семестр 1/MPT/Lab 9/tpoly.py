class TMember:
    """Класс для представления одночлена"""
    
    def __init__(self, c: int = 0, n: int = 0):
        """
        Конструктор одночлена
        c - коэффициент, n - степень
        """
        self._coeff = c
        self._degree = n if c != 0 else 0
    
    def get_degree(self) -> int:
        """ЧитатьСтепень - возвращает степень одночлена"""
        return self._degree
    
    def set_degree(self, n: int):
        """ПисатьСтепень - устанавливает степень одночлена"""
        self._degree = n
        if self._coeff == 0:
            self._degree = 0
    
    def get_coeff(self) -> int:
        """ЧитатьКоэффициент - возвращает коэффициент одночлена"""
        return self._coeff
    
    def set_coeff(self, c: int):
        """ПисатьКоэффициент - устанавливает коэффициент одночлена"""
        self._coeff = c
        if c == 0:
            self._degree = 0
    
    def __eq__(self, other: 'TMember') -> bool:
        """Равно - сравнение одночленов"""
        if not isinstance(other, TMember):
            return False
        return self._coeff == other._coeff and self._degree == other._degree
    
    def differentiate(self) -> 'TMember':
        """Дифференцировать - производная одночлена"""
        if self._degree == 0:
            return TMember(0, 0)
        return TMember(self._coeff * self._degree, self._degree - 1)
    
    def evaluate(self, x: float) -> float:
        """Вычислить - вычисление значения одночлена в точке x"""
        return self._coeff * (x ** self._degree)
    
    def to_string(self) -> str:
        """ОдночленВСтроку - строковое представление одночлена"""
        if self._coeff == 0:
            return "0"
        
        result = ""
        
        if self._coeff != 1 or self._degree == 0:
            if self._coeff == -1 and self._degree != 0:
                result += "-"
            else:
                result += str(self._coeff)
        
        if self._degree > 0:
            result += "x"
            if self._degree > 1:
                result += f"^{self._degree}"
        
        return result
    
    def __str__(self):
        return self.to_string()


class TPoly:
    """Класс для представления полинома"""
    
    def __init__(self, c: int = 0, n: int = 0):
        """
        Конструктор полинома
        c - коэффициент, n - степень для создания одночленного полинома
        """
        self._members = []
        if c != 0:
            self._members.append(TMember(c, n))
        self._normalize()
    
    def _normalize(self):
        """Приведение полинома к нормализованному виду"""
        if not self._members:
            return
        
        self._members.sort(key=lambda m: m.get_degree(), reverse=True)
        
        i = 0
        while i < len(self._members) - 1:
            current = self._members[i]
            next_member = self._members[i + 1]
            
            if current.get_degree() == next_member.get_degree():
                new_coeff = current.get_coeff() + next_member.get_coeff()
                if new_coeff != 0:
                    self._members[i] = TMember(new_coeff, current.get_degree())
                    del self._members[i + 1]
                else:
                    del self._members[i]
                    del self._members[i]
            else:
                i += 1
        
        self._members = [m for m in self._members if m.get_coeff() != 0]
    
    def get_degree(self) -> int:
        """Степень - возвращает степень полинома"""
        if not self._members:
            return 0
        return max(m.get_degree() for m in self._members)
    
    def get_coeff(self, n: int) -> int:
        """Коэффициент - возвращает коэффициент при степени n"""
        if not self._members:
            return 0
        for member in self._members:
            if member.get_degree() == n:
                return member.get_coeff()
        return 0
    
    def clear(self):
        """Очистить - удаляет все члены полинома"""
        self._members = []
    
    def add(self, other: 'TPoly') -> 'TPoly':
        """Сложить - сложение полиномов"""
        result = TPoly()
        result._members = self._members.copy()
        for member in other._members:
            result._members.append(TMember(member.get_coeff(), member.get_degree()))
        result._normalize()
        return result
    
    def multiply(self, other: 'TPoly') -> 'TPoly':
        """Умножить - умножение полиномов"""
        result = TPoly()
        for m1 in self._members:
            for m2 in other._members:
                new_coeff = m1.get_coeff() * m2.get_coeff()
                new_degree = m1.get_degree() + m2.get_degree()
                result._members.append(TMember(new_coeff, new_degree))
        result._normalize()
        return result
    
    def subtract(self, other: 'TPoly') -> 'TPoly':
        """Вычесть - вычитание полиномов"""
        return self.add(other.negate())
    
    def negate(self) -> 'TPoly':
        """Минус - унарный минус"""
        result = TPoly()
        for member in self._members:
            result._members.append(TMember(-member.get_coeff(), member.get_degree()))
        result._normalize()
        return result
    
    def __eq__(self, other: 'TPoly') -> bool:
        """Равно - сравнение полиномов"""
        if not isinstance(other, TPoly):
            return False
        if len(self._members) != len(other._members):
            return False
        for m1, m2 in zip(self._members, other._members):
            if m1 != m2:
                return False
        return True
    
    def differentiate(self) -> 'TPoly':
        """Дифференцировать - производная полинома"""
        result = TPoly()
        for member in self._members:
            if member.get_degree() > 0:
                derivative = member.differentiate()
                result._members.append(derivative)
        result._normalize()
        return result
    
    def evaluate(self, x: float) -> float:
        """Вычислить - вычисление значения полинома в точке x"""
        result = 0.0
        for member in self._members:
            result += member.evaluate(x)
        return result
    
    def get_member(self, i: int) -> tuple:
        """Элемент - доступ к члену полинома по индексу"""
        if i < 0 or i >= len(self._members):
            raise IndexError("Индекс выходит за границы")
        member = self._members[i]
        return member.get_coeff(), member.get_degree()
    
    def to_string(self) -> str:
        """Строковое представление полинома"""
        if not self._members:
            return "0"
        
        result = ""
        for i, member in enumerate(self._members):
            member_str = member.to_string()
            if i > 0:
                if member_str.startswith('-'):
                    result += " - " + member_str[1:]
                else:
                    result += " + " + member_str
            else:
                result += member_str
        
        return result
    
    def __str__(self):
        return self.to_string()
    
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.subtract(other)
    
    def __mul__(self, other):
        return self.multiply(other)
    
    def __neg__(self):
        return self.negate()


if __name__ == "__main__":
    print("Тестирование класса TMember:")
    m1 = TMember(3, 2)
    m2 = TMember(-1, 3)
    m3 = TMember(0, 5)
    print(f"m1 = {m1}")
    print(f"m2 = {m2}")
    print(f"m3 = {m3}")
    
    print(f"Степень m1: {m1.get_degree()}")
    print(f"Коэффициент m1: {m1.get_coeff()}")
    m1.set_coeff(5)
    m1.set_degree(4)
    print(f"После изменения: {m1}")
    
    m4 = TMember(4, 3)
    print(f"Производная {m4}: {m4.differentiate()}")
    
    print(f"m4(2) = {m4.evaluate(2)}")
    
    print("\nТестирование класса TPoly:")
    p1 = TPoly(2, 3)
    p2 = TPoly(3, 1)
    p3 = TPoly(5, 0)
    p4 = TPoly()
    print(f"p1 = {p1}")
    print(f"p2 = {p2}")
    print(f"p3 = {p3}")
    print(f"p4 = {p4}")
    
    p5 = p1.add(p2)
    print(f"p1 + p2 = {p5}")
    
    p6 = TPoly(1, 1)
    p7 = TPoly(1, 1)
    p8 = p6.multiply(p7)
    print(f"x * x = {p8}")
    
    p9 = p5.subtract(p1)
    print(f"(2x^3 + 3x) - 2x^3 = {p9}")
    
    p10 = p1.negate()
    print(f"-p1 = {p10}")
    
    p11 = TPoly(3, 1)
    print(f"p2 == p11: {p2 == p11}")
    
    p12 = TPoly(1, 3).add(TPoly(7, 1)).add(TPoly(5, 0))
    print(f"Производная {p12}: {p12.differentiate()}")
    
    p13 = TPoly(1, 2).add(TPoly(3, 1))
    print(f"p13(2) = {p13.evaluate(2)}")
    
    print("Члены полинома p12:")
    for i in range(len(p12._members)):
        coeff, degree = p12.get_member(i)
        print(f" Член {i}: коэффициент={coeff}, степень={degree}")

