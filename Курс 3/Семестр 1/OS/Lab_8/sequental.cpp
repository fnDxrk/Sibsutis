#include <ctime>
#include <iostream>

long long sequential_sum(long long N)
{
    volatile long long sum = 0;
    for (long long i = 0; i <= N; ++i) {
        sum += i;
    }
    return sum;
}

int main()
{
    long long N = 2e9;

    // Измерение времени выполнения
    clock_t start = clock();
    long long result = sequential_sum(N);
    clock_t end = clock();

    std::cout << "=========Sequential: " << std::endl;
    std::cout << "Сумма: " << result << std::endl;
    std::cout << "Время (последовательно): "
              << double(end - start) / CLOCKS_PER_SEC << " секунд" << std::endl;

    return 0;
}
