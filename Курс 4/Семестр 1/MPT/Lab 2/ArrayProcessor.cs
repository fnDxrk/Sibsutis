using System;

namespace ArrayOperations
{
  public class ArrayProcessor
  {
    public static int FindMinOfThree(int a, int b, int c)
    {
      if (a <= b && a <= c)
        return a;
      else if (b <= a && b <= c)
        return b;
      else
        return c;
    }

    public static double SumElementsWithEvenIndexSum(double[,] A)
    {
      if (A == null)
        throw new ArgumentNullException(nameof(A), "Массив не может быть null");

      double sum = 0;
      int rows = A.GetLength(0);
      int cols = A.GetLength(1);

      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < cols; j++)
        {
          if ((i + j) % 2 == 0)
          {
            sum += A[i, j];
          }
        }
      }

      return sum;
    }

    public static double MaxOnAndBelowMainDiagonal(double[,] A)
    {
      if (A == null)
        throw new ArgumentNullException(nameof(A), "Массив не может быть null");

      int rows = A.GetLength(0);
      int cols = A.GetLength(1);

      if (rows == 0 || cols == 0)
        throw new ArgumentException("Массив не может быть пустым");

      double max = double.MinValue;

      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j <= Math.Min(i, cols - 1); j++)
        {
          if (A[i, j] > max)
          {
            max = A[i, j];
          }
        }
      }

      return max;
    }
  }
}
