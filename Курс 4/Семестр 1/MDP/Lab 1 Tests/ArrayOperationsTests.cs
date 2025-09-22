using Lab1;

namespace Lab1Tests
{
    [TestClass]
    public class ArrayOperationsTests
    {
        // Тесты для SumEvenElements
        [TestMethod]
        public void SumEvenElements_BothEven_ReturnsSum()
        {
            // Проверяет успешное суммирование чётных элементов
            int[] a = { 2, 4 };
            int[] b = { 2, 6 };
            int[] expected = { 4, 10 };
            int[] result = ArrayOperations.SumEvenElements(a, b);
            CollectionAssert.AreEqual(expected, result);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void SumEvenElements_NullArray_ThrowsException()
        {
            // Проверяет выброс исключения при null
            ArrayOperations.SumEvenElements(null, new int[] { 1, 2 });
        }

        [TestMethod]
        public void SumEvenElements_NoEvenPairs_ReturnsEmpty()
        {
            // Проверяет случай без чётных пар
            int[] a = { 1, 3 };
            int[] b = { 1, 3 };
            int[] expected = { };
            int[] result = ArrayOperations.SumEvenElements(a, b);
            CollectionAssert.AreEqual(expected, result);
        }

        // Тесты для CyclicShiftLeft
        [TestMethod]
        public void CyclicShiftLeft_ValidShift_ShiftsArray()
        {
            // Проверяет сдвиг влево
            double[] array = { 1.0, 2.0, 3.0 };
            double[] expected = { 3.0, 1.0, 2.0 };
            ArrayOperations.CyclicShiftLeft(array, 2);
            CollectionAssert.AreEqual(expected, array);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void CyclicShiftLeft_NullArray_ThrowsException()
        {
            // Проверяет выброс исключения при null
            ArrayOperations.CyclicShiftLeft(null, 1);
        }

        [TestMethod]
        public void CyclicShiftLeft_EmptyArray_NoChange()
        {
            // Проверяет обработку пустого массива
            double[] array = { };
            double[] expected = { };
            ArrayOperations.CyclicShiftLeft(array, 5);
            CollectionAssert.AreEqual(expected, array);
        }

        // Тесты для FindSequenceIndex
        [TestMethod]
        public void FindSequenceIndex_SequenceFound_ReturnsIndex()
        {
            // Проверяет успешное нахождение последовательности
            int[] vec = { 1, 2, 3, 4 };
            int[] seq = { 2, 3 };
            int expected = 1;
            int result = ArrayOperations.FindSequenceIndex(vec, seq);
            Assert.AreEqual(expected, result);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FindSequenceIndex_NullVec_ThrowsException()
        {
            // Проверяет выброс исключения при null
            ArrayOperations.FindSequenceIndex(null, new int[] { 1 });
        }

        [TestMethod]
        public void FindSequenceIndex_NoSequence_ReturnsMinusOne()
        {
            // Проверяет случай, когда последовательность не найдена
            int[] vec = { 1, 2, 3 };
            int[] seq = { 4 };
            int expected = -1;
            int result = ArrayOperations.FindSequenceIndex(vec, seq);
            Assert.AreEqual(expected, result);
        }
    }
}