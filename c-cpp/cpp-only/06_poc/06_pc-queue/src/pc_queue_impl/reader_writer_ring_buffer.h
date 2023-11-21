// https://github.com/cameron314/readerwriterqueue

#include "../pc_queue.h"

#include <chrono>
#include <iostream>
#include <readerwriterqueue/readerwritercircularbuffer.h>

using brwcb = moodycamel::BlockingReaderWriterCircularBuffer<uint32_t *>;

template <>
inline PcQueue<brwcb>::PcQueue(size_t iterations, uint32_t *element_counter)
    : q(1024 * 1024 * 1024) {
  this->element_counter = element_counter;
  this->iterations = iterations;
}

template <> inline void PcQueue<brwcb>::consume() {
  uint32_t *msg;
  while (true) {
    if (msg_count < iterations) [[likely]] {
      if (q.wait_dequeue_timed(msg, std::chrono::nanoseconds(1000)))
          [[likely]] {
        ++msg_count;
        ++element_counter[*msg];
      }
    } else [[unlikely]] {
      break;
    }
  }
}

template <> inline void PcQueue<brwcb>::enqueue(uint32_t *msg) {
  if (!q.try_enqueue(msg)) {
    std::cerr << "q.try_enqueue(msg) false!" << std::endl;
  }
}
