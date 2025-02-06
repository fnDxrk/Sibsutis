#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>

class CircularBuffer {
public:
    CircularBuffer(size_t size) : buffer(size), head(0), tail(0), count(0) {}

    void produce(int item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_full.wait(lock, [this] { return count < buffer.size(); });

        buffer[head] = item;
        head = (head + 1) % buffer.size();
        ++count;

        std::cout << "Produced: " << item << std::endl;

        lock.unlock();
        cv_empty.notify_all();
    }

    int consume() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_empty.wait(lock, [this] { return count > 0; });

        int item = buffer[tail];
        tail = (tail + 1) % buffer.size();
        --count;

        std::cout << "Consumed: " << item << std::endl;

        lock.unlock();
        cv_full.notify_all();

        return item;
    }

private:
    std::vector<int> buffer;
    size_t head, tail, count;
    std::mutex mtx;
    std::condition_variable cv_empty;
    std::condition_variable cv_full;
};

void producer(CircularBuffer& cb) {
    for (int i = 0; i < 20; ++i) {
        cb.produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer(CircularBuffer& cb) {
    while (true) {
        cb.consume();
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

int main() {
    CircularBuffer cb(5);

    std::thread prod(producer, std::ref(cb));
    std::thread cons(consumer, std::ref(cb));

    prod.join();
    cons.join();

    return 0;
}
