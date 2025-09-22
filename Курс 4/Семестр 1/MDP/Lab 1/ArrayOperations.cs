namespace Lab1
{
    // Вариант 2
    public class ArrayOperations
    {
        /* Функция получает два одномерных целочисленных массив a, b
        одинаковой длины. Возвращает массив, полученный суммированием
        компонентов массивов a и b с чётными значениями. */
        public static int[] SumEvenElements(int[] a, int[] b)
        {
            if (a == null || b == null || a.Length != b.Length)
                throw new ArgumentException("Массивы должны быть ненулевыми и одинаковой длины!");

            List<int> sum = new List<int>();

            for (int i = 0; i < a.Length; i++)
                if (a[i] % 2 == 0 && b[i] % 2 == 0)
                    sum.Add(a[i] + b[i]);

            return sum.ToArray();
        }

        /* Функция получает одномерный массив вещественных переменных и
        целое – параметр сдвига. Функция изменяет массив циклическим сдвигом
        значений его элементов влево на число позиций, равное параметру сдвига. */
        public static void CyclicShiftLeft(double[] array, int shift)
        {
            int arraySize = array.Length;
            shift = shift % arraySize;

            if (shift > 0)
                shift = -shift;

            for (int i = 0; i > shift; i--)
            {
                double tmp = array[0];

                for (int j = 1; j < arraySize; j++)
                {
                    array[j - 1] = array[j];
                }
                array[arraySize - 1] = tmp;
            }
        }
        /* Функция находит и возвращает индекс начала первого вхождения
        последовательности целых чисел, представленных массивом int[] seq в
        другую последовательность, представленную массивом int[] vec. */
        public static int FirstSequenceIndex(int[] vec, int[] seq)
        {
            if (vec == null || seq == null)
                throw new ArgumentNullException("Массивы не должны быть нулевыми!");
            if (seq.Length == 0 || seq.Length > vec.Length)
                return -1;

            for (int i = 0; i <= vec.Length - seq.Length; i++)
            {
                bool found = true;
                for (int j = 0; j < seq.Length; j++)
                {
                    if (vec[i + j] != seq[j])
                    {
                        found = false;
                        break;
                    }
                }
                if (found)
                    return i;
            }

            return -1;
        }
    }
}