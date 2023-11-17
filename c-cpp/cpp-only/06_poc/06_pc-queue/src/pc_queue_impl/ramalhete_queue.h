// https://github.com/mpoeter/xenium

#include "../pc_queue.h"

#include <xenium/ramalhete_queue.hpp>
#include <xenium/reclamation/generic_epoch_based.hpp>

using Reclaimer =
    xenium::policy::reclaimer<xenium::reclamation::generic_epoch_based<>>;
using rq = xenium::ramalhete_queue<uint32_t *, Reclaimer,
                                   xenium::policy::entries_per_node<2048>>;

template <> void PcQueue<rq>::consume() {
  uint32_t *msg;
  while (true) {
    if (msg_count < iterations) [[likely]] {
      if (q.try_pop(msg)) [[likely]] {
        ++msg_count;
        ++element_counter[*msg];
      }
    } else [[unlikely]] {
      break;
    }
  }
}

template <> void PcQueue<rq>::enqueue(uint32_t *msg) { q.push(msg); }
