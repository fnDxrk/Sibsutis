#include "pch.h"
#include "CppUnitTest.h"
#include "ArrayProcessor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ArrayProcessorTests {
    TEST_CLASS(ArrayProcessorTests) {
    public:
        
        // ===== FormOddDigitsReverse Tests =====
        
        TEST_METHOD(FormOddDigitsReverse_AllOddDigits) {
            int result = ArrayProcessor::FormOddDigitsReverse(13579);
            Assert::AreEqual(97531, result);
        }
        
        TEST_METHOD(FormOddDigitsReverse_MixedDigits) {
            int result = ArrayProcessor::FormOddDigitsReverse(123456);
            Assert::AreEqual(531, result);
        }
        
        TEST_METHOD(FormOddDigitsReverse_AllEvenDigits) {
            int result = ArrayProcessor::FormOddDigitsReverse(2468);
            Assert::AreEqual(0, result);
        }
        
        TEST_METHOD(FormOddDigitsReverse_Zero) {
            int result = ArrayProcessor::FormOddDigitsReverse(0);
            Assert::AreEqual(0, result);
        }
        
        TEST_METHOD(FormOddDigitsReverse_NegativeNumber) {
            int result = ArrayProcessor::FormOddDigitsReverse(-12345);
            Assert::AreEqual(-531, result);
        }
        
        // ===== FindMaxEvenDigitInEvenPosition Tests =====
        
        TEST_METHOD(FindMaxEvenDigitInEvenPosition_ValidCase) {
            int result = ArrayProcessor::FindMaxEvenDigitInEvenPosition(62543);
            Assert::AreEqual(2, result);
        }
        
        TEST_METHOD(FindMaxEvenDigitInEvenPosition_NoEvenDigitsInEvenPositions) {
            int result = ArrayProcessor::FindMaxEvenDigitInEvenPosition(13579);
            Assert::AreEqual(-1, result);
        }
        
        TEST_METHOD(FindMaxEvenDigitInEvenPosition_SingleDigitEven) {
            int result = ArrayProcessor::FindMaxEvenDigitInEvenPosition(8);
            Assert::AreEqual(-1, result);
        }
        
        TEST_METHOD(FindMaxEvenDigitInEvenPosition_Zero) {
            int result = ArrayProcessor::FindMaxEvenDigitInEvenPosition(0);
            Assert::AreEqual(-1, result);
        }
        
        TEST_METHOD(FindMaxEvenDigitInEvenPosition_MultipleEvenDigits) {
            int result = ArrayProcessor::FindMaxEvenDigitInEvenPosition(246824);
            Assert::AreEqual(4, result);
        }
        
        TEST_METHOD(FindMaxEvenDigitInEvenPosition_NegativeNumber) {
            int result = ArrayProcessor::FindMaxEvenDigitInEvenPosition(-62543);
            Assert::AreEqual(2, result);
        }
        
        TEST_METHOD(FindMaxEvenDigitInEvenPosition_EvenDigitsInOddPositions) {
            int result = ArrayProcessor::FindMaxEvenDigitInEvenPosition(123456);
            Assert::AreEqual(-1, result);
        }
        
        // ===== CircularShiftRight Tests =====
        
        TEST_METHOD(CircularShiftRight_NormalCase) {
            int result = ArrayProcessor::CircularShiftRight(123456, 2);
            Assert::AreEqual(561234, result);
        }
        
        TEST_METHOD(CircularShiftRight_ZeroPositions) {
            int result = ArrayProcessor::CircularShiftRight(123456, 0);
            Assert::AreEqual(123456, result);
        }
        
        TEST_METHOD(CircularShiftRight_FullCycle) {
            int result = ArrayProcessor::CircularShiftRight(123456, 6);
            Assert::AreEqual(123456, result);
        }
        
        TEST_METHOD(CircularShiftRight_MoreThanDigits) {
            int result = ArrayProcessor::CircularShiftRight(123, 5);
            Assert::AreEqual(231, result);
        }
        
        TEST_METHOD(CircularShiftRight_SingleDigit) {
            int result = ArrayProcessor::CircularShiftRight(5, 3);
            Assert::AreEqual(5, result);
        }
        
        TEST_METHOD(CircularShiftRight_Zero) {
            int result = ArrayProcessor::CircularShiftRight(0, 2);
            Assert::AreEqual(0, result);
        }
        
        TEST_METHOD(CircularShiftRight_NegativeNumber) {
            int result = ArrayProcessor::CircularShiftRight(-123456, 2);
            Assert::AreEqual(-561234, result);
        }
        
        // ===== SumEvenAboveSecondaryDiagonal Tests =====
        
        TEST_METHOD(SumEvenAboveSecondaryDiagonal_ValidCase) {
            const int size = 3;
            int** A = new int*[size];
            
            for (int i = 0; i < size; i++) {
                A[i] = new int[size];
            }
            
            A[0][0] = 2; A[0][1] = 3; A[0][2] = 1;
            A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;
            A[2][0] = 8; A[2][1] = 7; A[2][2] = 9;
            
            int result = ArrayProcessor::SumEvenAboveSecondaryDiagonal(A, size);
            Assert::AreEqual(6, result);
            
            for (int i = 0; i < size; i++) {
                delete[] A[i];
            }
            delete[] A;
        }
        
        TEST_METHOD(SumEvenAboveSecondaryDiagonal_AllOdd) {
            const int size = 2;
            int** A = new int*[size];
            
            for (int i = 0; i < size; i++) {
                A[i] = new int[size];
            }
            
            A[0][0] = 1; A[0][1] = 3;
            A[1][0] = 5; A[1][1] = 7;
            
            int result = ArrayProcessor::SumEvenAboveSecondaryDiagonal(A, size);
            Assert::AreEqual(0, result);
            
            for (int i = 0; i < size; i++) {
                delete[] A[i];
            }
            delete[] A;
        }
        
        TEST_METHOD(SumEvenAboveSecondaryDiagonal_NullArray) {
            int result = ArrayProcessor::SumEvenAboveSecondaryDiagonal(nullptr, 3);
            Assert::AreEqual(0, result);
        }
    };
}
