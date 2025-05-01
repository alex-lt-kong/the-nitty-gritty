#include "./proofs-of-concept/cpp/09_lockfree/ringbuffer/ringbuffer-spsc-impl.h"
#include "2_unique-ptr-impl.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <print>
#include <random>
#include <thread>
#include <vector>

// Turn this on to examine result manually, or turn this off to check for
// potential race condition
constexpr bool debug_mode = false;

struct Message {
  size_t id;
  int value;
  explicit Message(int value) : value(value) {
    id = m_id_counter++;
    if constexpr (debug_mode)
      std::println("Message (id: {}) created, value: {}", id, value);
  }
  ~Message() {
    if constexpr (debug_mode)
      std::println("Message (id: {}, value: {})  destroyed", id, value);
  }

private:
  static inline size_t m_id_counter = 0;
};

int main() {
  size_t message_count = 0;
  if constexpr (debug_mode)
    message_count = 5;
  else
    message_count = 100'000'000;
  constexpr int consumer_count = 10;
  std::random_device rd;
  std::mt19937 gen(rd());
  using LockFreePcQueue =
      PoC::LockFree::RingBufferSPSC<std::shared_ptr<Message>>;
  // Using my_unique_ptr instead of std::unique_ptr
  std::vector<my_unique_ptr<LockFreePcQueue>> queues;
  for (int i = 0; i < consumer_count; i++) {
    queues.emplace_back(my_make_unique<LockFreePcQueue>(
        message_count > 100'000 ? 100'000 : message_count));
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto consumer = [&](int id, LockFreePcQueue *q, int max_delay_ms) {
    std::uniform_int_distribution dis(0, max_delay_ms);
    int64_t prev_msg_id = -1;
    if constexpr (!debug_mode)
      max_delay_ms = 0;
    while (true) {
      int delay = 0;
      if constexpr (debug_mode) {
        delay = dis(gen);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      }
      std::shared_ptr<Message> msg;
      if (!q->dequeue(msg)) {
        // std::println("Queue is empty!");
        continue;
      }
      if (msg->id != prev_msg_id + 1) {
        std::println(
            "Consumer {}: Message (id: {}, value: {}) is out of order, "
            "prev_msg_id: {}",
            id, msg->id, msg->value, prev_msg_id);
      }
      prev_msg_id = msg->id;
      if constexpr (debug_mode)
        std::println("Consumer {} processing Message (id: {}, value: {}) after "
                     "{}ms delay",
                     id, msg->id, msg->value, delay);
      if (msg->value == '\0')
        break;
    }
    std::println("Consumer {} exited as it reaches the end of the queue, "
                 "prev_msg_id: {}",
                 id, prev_msg_id);
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
    std::uniform_int_distribution dis(1, std::numeric_limits<int>::max());
    for (int i = 0; i < message_count; ++i) {
      // We also use NULL-termination to signal the end of the message series
      const auto msg_ptr =
          std::make_shared<Message>(i < message_count - 1 ? dis(gen) : '\0');
      if constexpr (debug_mode)
        std::println("Producing msg (id: {}, value: {})", msg_ptr->id,
                     msg_ptr->value);
      for (int j = 0; j < queues.size(); ++j) {
        // the SPSC queue's interface expects to take ownership...here we cant
        // give it...so we make a copy first... But we are only copying the
        // pointer, so hopefully it is not too bad
        auto msg_ptr_cpy = msg_ptr;
        // std::move() is fine, but we did the copy before...
        // https://stackoverflow.com/questions/41871115/why-would-i-stdmove-an-stdshared-ptr
        while (!queues[j]->enqueue(std::move(msg_ptr_cpy))) {
          std::println("queues[{}] is full (id: {}, value: {})", j, msg_ptr->id,
                       msg_ptr->value);
          std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
      }
    }
  };

  std::thread prod(producer);
  prod.join();

  for (auto &t : consumers) {
    t.join();
  }

  auto t1 = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  std::locale::global(std::locale("en_US.UTF-8"));
  std::println("Program exited gracefully, {:L} msg / sec",
               static_cast<size_t>(message_count / (elapsed / 1000.0)));
  return 0;
}
