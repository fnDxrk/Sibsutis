using Lab1;

namespace Lab1
{
    class Program
    {
        static void Main()
        {
            // int[] a = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            // int[] b = { 23, 48, 11, 12, 12, 32, 45, 56, 98 };
            // int[] task_1 = ArrayOperations.SumEvenElements(a, b);
            // foreach (int i in task_1)
            // {
            //     Console.WriteLine(i);
            // }

            double[] task_2 = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            ArrayOperations.CyclicShiftLeft(task_2, 2);
            foreach (double i in task_2)
            {
                Console.Write(i + " ");
            }
            Console.WriteLine();

            // int[] vec = { 1, 2, 3, 4, 5 };
            // int[] seq = { 2, 3 };
            // int task_3 = ArrayOperations.FindSequenceIndex(vec, seq);
            // Console.WriteLine(task_3);
        }
    }
}