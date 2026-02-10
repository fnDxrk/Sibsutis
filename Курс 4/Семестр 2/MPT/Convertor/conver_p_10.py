class Conver_P_10:
    
    @staticmethod
    def char_To_num(ch: str) -> int:
        if '0' <= ch <= '9':
            return ord(ch) - ord('0')
        elif 'A' <= ch <= 'F':
            return ord(ch) - ord('A') + 10
        elif 'a' <= ch <= 'f':
            return ord(ch) - ord('a') + 10
        else:
            raise ValueError(f"Недопустимый символ: '{ch}'")
    
    @staticmethod
    def convert(P_num: str, P: int, weight: float) -> float:
        result = 0.0
        for ch in P_num:
            digit = Conver_P_10.char_To_num(ch)
            if digit >= P:
                raise ValueError(f"Цифра '{ch}' недопустима для основания {P}")
            result += digit * weight
            weight /= P
        return result
    
    @staticmethod
    def dval(P_num: str, P: int) -> float:
        if P < 2 or P > 16:
            raise ValueError("Основание должно быть от 2 до 16")
        
        P_num = P_num.strip()
        
        sign = 1
        if P_num.startswith('-'):
            sign = -1
            P_num = P_num[1:]
        elif P_num.startswith('+'):
            P_num = P_num[1:]
        
        if '.' in P_num:
            int_part, frac_part = P_num.split('.', 1)
        else:
            int_part = P_num
            frac_part = ""
        
        if not int_part:
            int_part = "0"
        
        int_value = Conver_P_10.convert(int_part, P, P ** (len(int_part) - 1))
        
        frac_value = 0.0
        if frac_part:
            frac_value = Conver_P_10.convert(frac_part, P, 1.0 / P)
        
        return sign * (int_value + frac_value)

