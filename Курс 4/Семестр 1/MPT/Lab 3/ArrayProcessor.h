#ifndef ARRAY_PROCESSOR_H
#define ARRAY_PROCESSOR_H

class ArrayProcessor {
public:
    // Формирует число из нечетных цифр в обратном порядке
    static int FormOddDigitsReverse(int a);
    
    // Находит позицию максимальной четной цифры на четной позиции
    static int FindMaxEvenDigitInEvenPosition(int n);
    
    // Выполняет циклический сдвиг цифр числа вправо
    static int CircularShiftRight(int n, int positions);
    
    // Вычисляет сумму четных элементов выше побочной диагонали
    static int SumEvenAboveSecondaryDiagonal(int** A, int size);
};

#endif // ARRAY_PROCESSOR_H

