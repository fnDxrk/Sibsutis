class Matrix:
    def __init__(self, data):
        """
        Конструктор матрицы
        Вход: двумерный массив целых чисел
        Предусловия: число строк и столбцов > 0
        """
        if not data or len(data) == 0 or len(data[0]) == 0:
            raise ValueError("Матрица должна иметь хотя бы одну строку и столбец")
        # Проверяем, что все строки одинаковой длины
        row_length = len(data[0])
        for row in data:
            if len(row) != row_length:
                raise ValueError("Все строки матрицы должны быть одинаковой длины")
        self._data = [row[:] for row in data]  # Глубокая копия
        self._i = len(data)  # число строк
        self._j = len(data[0])  # число столбцов

    @property
    def I(self):
        """Число строк"""
        return self._i

    @property
    def J(self):
        """Число столбцов"""
        return self._j

    def __getitem__(self, indices):
        """
        Взять элемент с индексами i,j
        Предусловия: индексы в допустимых диапазонах
        """
        i, j = indices
        if not (0 <= i < self._i and 0 <= j < self._j):
            raise IndexError("Индексы выходят за границы матрицы")
        return self._data[i][j]

    def __add__(self, other):
        """
        Сложение матриц
        Предусловия: размеры матриц совпадают
        """
        if self._i != other._i or self._j != other._j:
            raise ValueError("Размеры матриц должны совпадать для сложения")
        result = []
        for i in range(self._i):
            row = []
            for j in range(self._j):
                row.append(self._data[i][j] + other._data[i][j])
            result.append(row)
        return Matrix(result)

    def __sub__(self, other):
        """
        Вычитание матриц
        Предусловия: размеры матриц совпадают
        """
        if self._i != other._i or self._j != other._j:
            raise ValueError("Размеры матриц должны совпадать для вычитания")
        result = []
        for i in range(self._i):
            row = []
            for j in range(self._j):
                row.append(self._data[i][j] - other._data[i][j])
            result.append(row)
        return Matrix(result)

    def __mul__(self, other):
        """
        Умножение матриц
        Предусловия: матрицы согласованы для умножения
        """
        if self._j != other._i:
            raise ValueError(
                "Число столбцов первой матрицы должно равняться числу строк второй матрицы"
            )
        result = []
        for i in range(self._i):
            row = []
            for j in range(other._j):
                sum_val = 0
                for k in range(self._j):
                    sum_val += self._data[i][k] * other._data[k][j]
                row.append(sum_val)
            result.append(row)
        return Matrix(result)

    def __eq__(self, other):
        """
        Проверка равенства матриц
        Предусловия: размеры матриц совпадают
        """
        if self._i != other._i or self._j != other._j:
            return False
        for i in range(self._i):
            for j in range(self._j):
                if self._data[i][j] != other._data[i][j]:
                    return False
        return True

    def transp(self):
        """
        Транспонирование матрицы
        Предусловия: матрица квадратная
        """
        if self._i != self._j:
            raise ValueError("Матрица должна быть квадратной для транспонирования")
        result = []
        for j in range(self._j):
            row = []
            for i in range(self._i):
                row.append(self._data[i][j])
            result.append(row)
        return Matrix(result)

    def min(self):
        """
        Поиск минимального элемента
        """
        if self._i == 0 or self._j == 0:
            raise ValueError("Матрица пустая")
        min_val = self._data[0][0]
        for i in range(self._i):
            for j in range(self._j):
                if self._data[i][j] < min_val:
                    min_val = self._data[i][j]
        return min_val

    def __str__(self):
        """
        Преобразование матрицы в строку
        """
        rows = []
        for row in self._data:
            rows.append("{" + ",".join(map(str, row)) + "}")
        return "{" + ",".join(rows) + "}"

    def __repr__(self):
        return self.__str__()
