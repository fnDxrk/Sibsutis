#include <atomic>
#include <iomanip>
#include <iostream>
#include <thread>
using namespace std;

const long long num_steps = 1000000000; // количество шагов для вычисления
const double step = 1.0 / static_cast<double>(num_steps);

std::atomic<double> sum { 0.0 };
std::atomic<int> flag { 0 };
std::atomic<int> turn { 0 };

void calculatePi(int id, int num_threads)
{
    double x;
    double partial_sum = 0.0;

    for (int i = id; i < num_steps; i += num_threads) {
        x = (i + 0.5) * step;
        partial_sum += 4.0 / (1.0 + x * x);
    }

    partial_sum *= step;

    // Вход в критическую секцию
    flag.store(id, memory_order_relaxed);
    turn.store(num_threads - 1 - id, memory_order_relaxed);

    for (int j = 0; j < num_threads; ++j) {
        while ((j != id) and (flag.load(memory_order_relaxed) == id) and (turn.load(memory_order_relaxed) == num_threads - 1 - id)) {
            // Ждем своей очереди
            this_thread::yield();
        }
    }

    sum.fetch_add(partial_sum, memory_order_relaxed);

    // Выход из критической секции
    flag.store(-1, memory_order_relaxed);
}

int main()
{
    int num_threads = thread::hardware_concurrency();
    // int num_threads = 1;
    cout << "num_threads = " << num_threads << endl;

    thread threads[num_threads];

    // Создание потоков для вычислений
    for (int i = 0; i < num_threads; ++i)
        threads[i] = thread(calculatePi, i, num_threads);

    // Ожидание завершения потоков
    for (int i = 0; i < num_threads; ++i)
        threads[i].join();

    cout << "Реальное число pi:    "
            "3.14159265358979323846264338327950288419716939937510582097494459"
         << endl;
    cout << setprecision(64)
         << "Вычисляемое число pi: " << sum.load(memory_order_relaxed) << endl;

    return 0;
}
