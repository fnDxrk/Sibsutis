using System;

int[][] findMinData = new int[][] {
  new int[] { 5, 2, 8 },
  new int[] { 5, 5, 5 },
  new int[] { 3, 3, 7 },
  new int[] { -5, -2, -8 },
  new int[] { 0, 5, -3 },
  new int[] { 10, 10, 5 }
};

double[][,] sumEvenIndexData = new double[][,] {
  new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
  new double[,] { { 5.5 } },
  new double[,] { { 1, 2 }, { 3, 4 } },
  new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
  new double[,] { { 0, 0 }, { 0, 0 } }
};

double[][,] maxDiagonalData = new double[][,] {
  new double[,] { { 10, 20, 30 }, { 40, 50, 60 }, { 70, 80, 90 } },
  new double[,] { { 1, 99, 99 }, { 100, 2, 99 }, { 50, 60, 3 } },
  new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } },
  new double[,] { { 7.7 } },
  new double[,] { { 5, 1 }, { 2, 4 } },
  new double[,] { { -5, -2 }, { -8, -1 } }
};

object[][] multiplyNonZeroData = new object[][] {
  new object[] { new double[] { 1, 2, 3, 4, 5 }, new int[] { 0, 2, 4 }, 15.0 },
  new object[] { new double[] { 1, 0, 3, 0, 5 }, new int[] { 0, 1, 2, 3, 4 }, 15.0 },
  new object[] { new double[] { 0, 0, 0 }, new int[] { 0, 1, 2 }, 0.0 },
  new object[] { new double[] { 2.5, 1.5, 4.0 }, new int[] { 0, 1, 2 }, 15.0 },
  new object[] { new double[] { -2, 3, -4 }, new int[] { 0, 2 }, 8.0 }
};

object[][] findMinElementData = new object[][] {
  new object[] { new int[] { 5, 2, 8, 1, 9 }, (1, 3) },
  new object[] { new int[] { 5, 5, 5, 5 }, (5, 0) },
  new object[] { new int[] { -5, -2, -8, -1 }, (-8, 2) },
  new object[] { new int[] { 0, 0, 0 }, (0, 0) },
  new object[] { new int[] { 15, 8, 23, 4, 42, 7 }, (4, 3) }
};

object[][] reverseArrayData = new object[][] {
  new object[] { new double[] { 1.1, 2.2, 3.3, 4.4 }, new double[] { 4.4, 3.3, 2.2, 1.1 } },
  new object[] { new double[] { 5.5 }, new double[] { 5.5 } },
  new object[] { new double[] {}, new double[] {} },
  new object[] { new double[] { -1.1, -2.2, -3.3 }, new double[] { -3.3, -2.2, -1.1 } },
  new object[] { new double[] { 1, 2, 3, 4, 5 }, new double[] { 5, 4, 3, 2, 1 } }
};

object[][] exceptionData = new object[][] {
  new object[] { null, new int[] { 0, 1 }, typeof(ArgumentNullException) },
  new object[] { new double[] { 1, 2, 3 }, null, typeof(ArgumentNullException) },
  new object[] { new double[] { 1, 2, 3 }, new int[] { 0, 5 }, typeof(IndexOutOfRangeException) },
  new object[] { new double[] { 1, 2, 3 }, new int[] { -1, 0 }, typeof(IndexOutOfRangeException) },
  new object[] { Array.Empty<int>(), typeof(ArgumentException) },
  new object[] { null, typeof(ArgumentNullException) }
};
