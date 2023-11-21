#ifndef PC_QUEUE_H
#define PC_QUEUE_H

#include <atomic>
#include <iostream>
#include <thread>

template <class T_QUEUE> class PcQueue {
private:
  T_QUEUE q;
  std::thread consumer;
  size_t iterations;
  size_t msg_count = 0;
  void consume();
  std::mutex mut;
  uint32_t *element_counter;

public:
  inline PcQueue(size_t iterations, uint32_t *element_counter) {
    this->element_counter = element_counter;
    this->iterations = iterations;
  }

  inline void start() { consumer = std::thread(&PcQueue::consume, this); }

  inline void wait() {
    if (consumer.joinable()) {
      consumer.join();
    }
  }

  inline ~PcQueue() { // wait();
  }

  void enqueue(uint32_t *msg);
  inline size_t handled_msg_count() { return this->msg_count; };
};

#endif // RS_PC_QUEUE_H
