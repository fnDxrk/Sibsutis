#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

std::mutex mtx;
std::condition_variable cv_empty;
std::condition_variable cv_full;
std::queue<int> buffer;
const size_t BUFFER_SIZE = 5;

void producer() {
    for (int i = 0; i < 20; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_full.wait(lock, [] { return buffer.size() < BUFFER_SIZE; });

        buffer.push(i);
        std::cout << "Produced: " << i << std::endl;

        lock.unlock();
        cv_empty.notify_all();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_empty.wait(lock, [] { return !buffer.empty(); });

        int item = buffer.front();
        buffer.pop();
        std::cout << "Consumed: " << item << std::endl;

        lock.unlock();
        cv_full.notify_all();

        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

int main() {
    std::thread prod(producer);
    std::thread cons(consumer);

    prod.join();
    cons.join();

    return 0;
}
