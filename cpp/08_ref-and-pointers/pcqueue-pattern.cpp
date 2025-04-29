#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <vector>
#include <print>     // C++23 printing header
#include <chrono>
#include <random>

// Data structure representing an incoming object.
struct Data {
    int value;
    Data(int v) : value(v) {
        std::println("Data {} created", value);
    }
    ~Data() {
        std::println("Data {} destroyed", value);
    }
};

// A simple thread-safe queue that holds std::shared_ptr<Data>.
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mtx_;
public:
    // Enqueue an item.
    void enqueue(T item) {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(item);
    }

    // Busy-wait dequeue: repeatedly polls until an item is available.
    T dequeue() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (!queue_.empty()) {
                    T item = queue_.front();
                    queue_.pop();
                    return item;
                }
            }
            // Yield to reduce the busy-wait overhead.
            std::this_thread::yield();
        }
    }
};

int main() {
    constexpr int numDataPoints = 5;
    constexpr int numConsumers = 4;

    // Create one queue per consumer thread.
    ThreadSafeQueue<std::shared_ptr<Data>> queues[numConsumers];

    // Consumer thread function with a maximum random delay parameter.
    auto consumer = [&](int id, ThreadSafeQueue<std::shared_ptr<Data>>& queue, int maxDelayMs) {
        // Set up a random engine for variable delays.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, maxDelayMs);

        for (int i = 0; i < numDataPoints; i++) {
            // Introduce a random delay before processing each item.
            int delay = dis(gen);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));

            auto data = queue.dequeue();
            std::println("Consumer {} processing Data {} after {}ms delay", id, data->value, delay);
        }
        std::println("Consumer {} finished processing.", id);
    };

    // Launch consumer threads with a maximum random delay of 150 ms.
    int maxDelayMs = 150;
    std::vector<std::thread> consumers;
    for (int i = 0; i < numConsumers; i++) {
        consumers.emplace_back(consumer, i, std::ref(queues[i]), maxDelayMs);
    }

    // Producer thread function: produces numDataPoints items.
    auto producer = [&]() {
        for (int i = 1; i <= numDataPoints; i++) {
            auto data = std::make_shared<Data>(i);
            std::println("Producer enqueuing Data {}", i);

            // Enqueue the same data pointer to all consumer queues.
            for (int j = 0; j < numConsumers; j++) {
                queues[j].enqueue(data);
            }
            // Simulate a delay between productions.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };

    std::thread prod(producer);
    prod.join();

    // Wait for all consumer threads to finish processing.
    for (auto& t : consumers) {
        t.join();
    }

    std::println("All consumers have finished. Exiting main.");
    return 0;
}
