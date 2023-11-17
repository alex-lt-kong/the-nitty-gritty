// https://github.com/cameron314/readerwriterqueue

#include "../pc_queue.h"

#include <readerwriterqueue/readerwriterqueue.h>

using rwq = moodycamel::ReaderWriterQueue<uint32_t *>;

template <> inline void PcQueue<rwq>::consume() {
  uint32_t *msg;
  while (true) {
    if (msg_count < iterations) [[likely]] {
      if (q.try_dequeue(msg)) [[likely]] {
        ++msg_count;
        ++element_counter[*msg];
      }
    } else [[unlikely]] {
      break;
    }
  }
}

template <> inline void PcQueue<rwq>::enqueue(uint32_t *msg) { q.enqueue(msg); }
