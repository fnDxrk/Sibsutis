#include <atomic>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

const int iterations_per_thread = 125000000; // Количество итераций на каждом потоке

atomic<double> pi { 0.0 }; // Используем атомарную переменную для суммирования
atomic_flag lock1 = ATOMIC_FLAG_INIT; // Используем atomic_flag для спин-блокировки

void calculatePi(int start, int end)
{
    double localSum = 0;
    int j = 2 * start + 1;

    for (int i = start; i < end; i++) {
        if (i % 2 == 0) {
            localSum += 4.0 / j;
        } else {
            localSum -= 4.0 / j;
        }
        j += 2;
    }

    while (lock1.test_and_set(memory_order_acquire)) {
    } // Захватываем блокировку

    pi = pi + localSum;

    lock1.clear(memory_order_release); // Освобождаем блокировку
}

int main()
{
    int num_threads = thread::hardware_concurrency();
    cout << "num_threads = " << num_threads << endl;
    // int num_threads = 1;

    vector<thread> threads;

    for (int i = 0; i < num_threads; i++) {
        int start = i * iterations_per_thread;
        int end = start + iterations_per_thread;
        threads.push_back(thread(calculatePi, start, end));
    }

    for (auto& t : threads)
        t.join();
    cout << "Реальное число pi:    "
            "3.14159265358979323846264338327950288419716939937510582097494459"
         << endl;
    cout << setprecision(64) << "Вычисляемое число pi: " << pi << endl;

    return 0;
}
