#include "../pc_queue.h"

#include <deque>
#include <mutex>

using dq = std::deque<uint32_t *>;

template <> inline void PcQueue<dq>::consume() {
  uint32_t *msg;
  while (true) {
    std::lock_guard<std::mutex> lk(mut);
    if (msg_count < iterations) [[likely]] {
      if (!q.empty()) [[likely]] {
        msg = q.front();
        q.pop_front();
        ++msg_count;
        ++element_counter[*msg];
      }
    } else [[unlikely]] {
      break;
    }
  }
}

template <> inline void PcQueue<dq>::enqueue(uint32_t *msg) {
  std::lock_guard<std::mutex> lk(mut);
  q.push_back(msg);
}
