#include "./proofs-of-concept/cpp/09_lockfree/ringbuffer/ringbuffer-spsc-impl.h"

#include <chrono>
#include <memory>
#include <print>
#include <random>
#include <thread>
#include <vector>

struct Data {
  int id;
  int value;
  explicit Data(int value) : value(value) {
    id = m_id_counter++;
    std::println("Data (id: {}) created, value: {}", id, value);
  }
  ~Data() { std::println("Data (id: {})  destroyed", id); }

private:
  static inline int m_id_counter = 0;
};

int main() {
  constexpr int numDataPoints = 5;
  constexpr int consumer_count = 4;

  using LockFreePcQueue = PoC::LockFree::RingBufferSPSC<std::shared_ptr<Data>>;
  std::vector<std::unique_ptr<LockFreePcQueue>> queues;
  for (int i = 0; i < consumer_count; i++) {
    queues.emplace_back(std::make_unique<LockFreePcQueue>(1024));
  }

  auto consumer = [&](int id, LockFreePcQueue *q, const int max_delay_ms) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution dis(0, max_delay_ms);

    for (int i = 0; i < numDataPoints; i++) {
      // Introduce a random delay before processing each item.
      int delay = dis(gen);
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      std::shared_ptr<Data> data;
      const auto res = q->dequeue(data);
      if (!res)
        continue;
      std::println("Consumer {} processing Data {} after {}ms delay", id,
                   data->value, delay);
    }
    std::println("Consumer {} finished processing.", id);
  };

  std::vector<std::thread> consumers;
  for (int i = 0; i < consumer_count; i++) {
    constexpr int max_delay_ms = 150;
    consumers.emplace_back(consumer, i,
                           // tricky: we cant std::move() and also we cant just
                           // make a copy, and so we pass a raw pointer...
                           queues[i].get(),
                           max_delay_ms * (consumer_count - i));
  }

  auto producer = [&] {
    for (int i = 1; i <= numDataPoints; i++) {
      const auto data = std::make_shared<Data>(i);
      std::println("Producer enqueuing Data {}", i);
      for (int j = 0; j < consumer_count; j++) {
        // the SPSC queue's interface expects to take ownership...here we cant
        // give it...so we make a copy first...
        auto data_cpy = data;
        // std::move() is fine, but we did the copy before...
        // https://stackoverflow.com/questions/41871115/why-would-i-stdmove-an-stdshared-ptr
        queues[j]->enqueue(std::move(data_cpy));
      }
    }
  };

  std::thread prod(producer);
  prod.join();

  for (auto &t : consumers) {
    t.join();
  }

  return 0;
}
