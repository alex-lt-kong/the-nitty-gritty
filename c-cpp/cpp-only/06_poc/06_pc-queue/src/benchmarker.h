#ifndef BENCHMARKER_H
#define BENCHMARKER_H

#include "pc_queue.h"

#include <chrono>

template <class T_QUEUE> auto benchmark() {
  auto start = std::chrono::high_resolution_clock::now();
  {
    PcQueue<T_QUEUE> pcq;
    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < 1000000; i++) {
      pcq.enqueue(i);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

#endif // BENCHMARKER_H