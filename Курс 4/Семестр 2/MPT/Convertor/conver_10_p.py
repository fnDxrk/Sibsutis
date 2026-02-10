class Conver_10_P:
    
    @staticmethod
    def int_to_Char(d: int) -> str:
        if 0 <= d <= 9:
            return str(d)
        elif 10 <= d <= 15:
            return chr(ord('A') + d - 10)
        else:
            raise ValueError(f"Цифра {d} вне диапазона 0..15")
    
    @staticmethod
    def int_to_P(n: int, p: int) -> str:
        if n == 0:
            return "0"
        
        result = ""
        n = abs(n)
        
        while n > 0:
            digit = n % p
            result = Conver_10_P.int_to_Char(digit) + result
            n = n // p
        
        return result
    
    @staticmethod
    def flt_to_P(x: float, p: int, c: int) -> str:
        result = ""
        x = abs(x)
        
        for _ in range(c):
            x = x * p
            digit = int(x)
            result += Conver_10_P.int_to_Char(digit)
            x = x - digit
        
        return result
    
    @staticmethod
    def Do(n: float, p: int, c: int) -> str:
        if p < 2 or p > 16:
            raise ValueError("Основание должно быть от 2 до 16")
        
        sign = "-" if n < 0 else ""
        n = abs(n)
        
        integer_part = int(n)
        fractional_part = n - integer_part
        
        int_str = Conver_10_P.int_to_P(integer_part, p)
        
        if fractional_part > 0:
            frac_str = Conver_10_P.flt_to_P(fractional_part, p, c)
            frac_str = frac_str.rstrip('0')
            if frac_str:
                return f"{sign}{int_str}.{frac_str}"
        
        return f"{sign}{int_str}"

