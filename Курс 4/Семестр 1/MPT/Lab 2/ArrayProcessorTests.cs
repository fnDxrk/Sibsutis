using System;
using Xunit;
using ArrayOperations;

namespace TestProject1
{
  public class UnitTest1
  {
    // Тесты для FindMinOfThree
    [Fact]
    public void FindMinOfThree_AllDifferent_ReturnsMin()
    {
      // Arrange
      int a = 5, b = 2, c = 8;

      // Act
      int result = ArrayProcessor.FindMinOfThree(a, b, c);

      // Assert
      Assert.Equal(2, result);
    }

    [Fact]
    public void FindMinOfThree_AllEqual_ReturnsSame()
    {
      // Arrange
      int a = 5, b = 5, c = 5;

      // Act
      int result = ArrayProcessor.FindMinOfThree(a, b, c);

      // Assert
      Assert.Equal(5, result);
    }

    [Fact]
    public void FindMinOfThree_TwoEqualMin_ReturnsMin()
    {
      // Arrange
      int a = 3, b = 3, c = 7;

      // Act
      int result = ArrayProcessor.FindMinOfThree(a, b, c);

      // Assert
      Assert.Equal(3, result);
    }

    [Fact]
    public void FindMinOfThree_NegativeNumbers_ReturnsMin()
    {
      // Arrange
      int a = -5, b = -2, c = -8;

      // Act
      int result = ArrayProcessor.FindMinOfThree(a, b, c);

      // Assert
      Assert.Equal(-8, result);
    }

    // Тесты для SumElementsWithEvenIndexSum
    [Fact]
    public void SumElementsWithEvenIndexSum_ValidArray_ReturnsCorrectSum()
    {
      // Arrange
      double[,] A = {
      { 1, 2, 3 },
      { 4, 5, 6 },
      { 7, 8, 9 }
    };

      // Act
      double result = ArrayProcessor.SumElementsWithEvenIndexSum(A);

      // Assert
      Assert.Equal(1 + 3 + 5 + 7 + 9, result); // (0,0)+(0,2)+(1,1)+(2,0)+(2,2)
    }

    [Fact]
    public void SumElementsWithEvenIndexSum_SingleElement_ReturnsElement()
    {
      // Arrange
      double[,] A = { { 5.5 } };

      // Act
      double result = ArrayProcessor.SumElementsWithEvenIndexSum(A);

      // Assert
      Assert.Equal(5.5, result);
    }

    [Fact]
    public void SumElementsWithEvenIndexSum_EmptyArray_ReturnsZero()
    {
      // Arrange
      double[,] A = new double[0, 0];

      // Act
      double result = ArrayProcessor.SumElementsWithEvenIndexSum(A);

      // Assert
      Assert.Equal(0, result);
    }

    [Fact]
    public void SumElementsWithEvenIndexSum_NullArray_ThrowsException()
    {
      // Arrange
      double[,] A = null;

      // Act & Assert
      Assert.Throws<ArgumentNullException>(() => ArrayProcessor.SumElementsWithEvenIndexSum(A));
    }

    // Тесты для MaxOnAndBelowMainDiagonal
    [Fact]
    public void MaxOnAndBelowMainDiagonal_ValidArray_ReturnsCorrectMax()
    {
      // Arrange
      double[,] A = {
      { 10, 20, 30 },
      { 40, 50, 60 },
      { 70, 80, 90 }
    };

      // Act
      double result = ArrayProcessor.MaxOnAndBelowMainDiagonal(A);

      // Assert
      Assert.Equal(90, result); // Максимум среди 10,40,50,70,80,90
    }

    [Fact]
    public void MaxOnAndBelowMainDiagonal_MaxBelowDiagonal_ReturnsCorrectMax()
    {
      // Arrange
      double[,] A = {
      { 1, 99, 99 },
      { 100, 2, 99 },
      { 50, 60, 3 }
    };

      // Act
      double result = ArrayProcessor.MaxOnAndBelowMainDiagonal(A);

      // Assert
      Assert.Equal(100, result); // Максимум ниже диагонали
    }

    [Fact]
    public void MaxOnAndBelowMainDiagonal_NonSquareArray_ReturnsCorrectMax()
    {
      // Arrange
      double[,] A = {
      { 1, 2, 3, 4 },
      { 5, 6, 7, 8 },
      { 9, 10, 11, 12 }
    };

      // Act
      double result = ArrayProcessor.MaxOnAndBelowMainDiagonal(A);

      // Assert
      Assert.Equal(11, result); // Максимум среди 1,5,6,9,10,11
    }

    [Fact]
    public void MaxOnAndBelowMainDiagonal_SingleElement_ReturnsElement()
    {
      // Arrange
      double[,] A = { { 7.7 } };

      // Act
      double result = ArrayProcessor.MaxOnAndBelowMainDiagonal(A);

      // Assert
      Assert.Equal(7.7, result);
    }

    [Fact]
    public void MaxOnAndBelowMainDiagonal_EmptyArray_ThrowsException()
    {
      // Arrange
      double[,] A = new double[0, 0];

      // Act & Assert
      Assert.Throws<ArgumentException>(() => ArrayProcessor.MaxOnAndBelowMainDiagonal(A));
    }

    [Fact]
    public void MaxOnAndBelowMainDiagonal_NullArray_ThrowsException()
    {
      // Arrange
      double[,] A = null;

      // Act & Assert
      Assert.Throws<ArgumentNullException>(() => ArrayProcessor.MaxOnAndBelowMainDiagonal(A));
    }
  }
}
