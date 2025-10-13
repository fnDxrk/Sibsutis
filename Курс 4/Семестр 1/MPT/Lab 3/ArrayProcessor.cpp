#include "ArrayProcessor.h"
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>

int ArrayProcessor::FormOddDigitsReverse(int a) {
    if (a == 0) return 0;
    
    int result = 0;
    int temp = abs(a);
    
    while (temp > 0) {
        int digit = temp % 10;
        if (digit % 2 != 0) {
            result = result * 10 + digit;
        }
        temp /= 10;
    }
    
    return (a < 0) ? -result : result;
}

int ArrayProcessor::FindMaxEvenDigitInEvenPosition(int n) {
    if (n == 0) {
        return -1;
    }
    
    int temp = abs(n);
    
    // Дополнительная страховка
    if (temp == 0) {
        return -1;
    }
    
    int position = 1;
    int maxDigit = -1;
    int maxPosition = -1;
    
    while (temp > 0) {
        int digit = temp % 10;
        
        if (position % 2 == 0 && digit % 2 == 0) {
            if (maxPosition == -1 || digit >= maxDigit) {
                maxDigit = digit;
                maxPosition = position;
            }
        }
        
        temp /= 10;
        position++;
    }
    
    // Явно возвращаем -1, если ничего не нашли
    return (maxPosition == -1) ? -1 : maxPosition;
}

int ArrayProcessor::CircularShiftRight(int n, int positions) {
    if (n == 0) return 0;
    
    int temp = abs(n);
    std::string numStr = std::to_string(temp);
    int numDigits = numStr.length();
    
    if (numDigits <= 0) return 0;
    
    positions = positions % numDigits;
    if (positions == 0) return n;
    
    // Циклический сдвиг ВПРАВО на positions позиций
    std::string rightPart = numStr.substr(numDigits - positions);
    std::string leftPart = numStr.substr(0, numDigits - positions);
    std::string resultStr = rightPart + leftPart;
    
    try {
        int result = std::stoi(resultStr);
        return (n < 0) ? -result : result;
    }
    catch (...) {
        return 0;
    }
}

int ArrayProcessor::SumEvenAboveSecondaryDiagonal(int** A, int size) {
    if (A == nullptr || size <= 0) return 0;
    
    int sum = 0;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (A[i][j] % 2 == 0) {
                sum += A[i][j];
            }
        }
    }
    
    return sum;
}

