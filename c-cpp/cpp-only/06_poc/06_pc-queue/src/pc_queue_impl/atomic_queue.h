// https://github.com/cameron314/readerwriterqueue

#include "../pc_queue.h"

#include <atomic_queue/atomic_queue.h>

using Element = uint32_t *;
inline uint32_t NIL = 0;
// Element constexpr NIL = static_cast<Element>(&a);
using aqb = atomic_queue::AtomicQueue<Element, 2048, &NIL>;

template <> inline void PcQueue<aqb>::consume() {
  uint32_t *msg;
  while (true) {
    if (msg_count < iterations) [[likely]] {
      // What if q is empty()? The doc doesn't make it very clear...
      msg = q.pop();
      ++msg_count;
      ++element_counter[*msg];
    } else [[unlikely]] {
      break;
    }
  }
}

template <> inline void PcQueue<aqb>::enqueue(uint32_t *msg) { q.push(msg); }
