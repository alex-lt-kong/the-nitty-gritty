#include "pc_queue.h"
#include "pc_queue_impl/atomic_queue.h"
#include "pc_queue_impl/mutex_deque.h"
#include "pc_queue_impl/ramalhete_queue.h"
#include "pc_queue_impl/reader_writer_queue.h"

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

using namespace std;

template <class T_QUEUE>
inline auto benchmark(size_t iterations, uint32_t *elements,
                      size_t element_count, uint32_t *element_counter) {
  auto start = chrono::high_resolution_clock::now();

  auto pcq = PcQueue<T_QUEUE>(iterations, element_counter);
  pcq.start();
  start = chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < iterations; i++) {
    pcq.enqueue(&elements[i % element_count]);
  }
  pcq.wait();
  auto end = chrono::high_resolution_clock::now();

  return std::make_tuple(
      pcq.handled_msg_count(),
      chrono::duration_cast<chrono::nanoseconds>(end - start).count());
}

template <class T_QUEUE> void benchmark_executor(string impl_name) {
  constexpr size_t iter_count = 1000 * 1000 * 100;
  uint32_t ele_arr[] = {0, 2, 2, 2, 4, 5, 5, 7, 8, 9};
  constexpr size_t ele_len = sizeof(ele_arr) / sizeof(ele_arr[0]);

  cout << "===== " << impl_name << " =====\n";

  for (size_t i = 0; i < 10; ++i) {
    uint32_t ele_counter[ele_len] = {0};
    auto [msg_count, elasped_ns] =
        benchmark<T_QUEUE>(iter_count, ele_arr, ele_len, ele_counter);
    std::locale loc("");
    std::cout.imbue(loc);
    cout << "iter: " << i << ", elasped_ms: " << elasped_ns / 1000 / 1000
         << ", handled_msg: " << msg_count
         << ", ops/sec: " << msg_count * 1000 * 1000 * 1000 / elasped_ns
         << ", counter: ";
    for (size_t j = 0; j < 10; ++j) {
      cout << ele_counter[j];
      if (j < 9)
        cout << "|";
    }
    cout << "\n";
  }
  std::cout << "\n" << std::endl;
}

void print_cpu_model() {
  char buffer[PATH_MAX];
  FILE *fp = fopen("/proc/cpuinfo", "r");

  if (fp == NULL) {
    perror("Failed to open /proc/cpuinfo");
    return;
  }

  while (fgets(buffer, PATH_MAX, fp) != NULL) {
    if (strncmp(buffer, "model name", 10) == 0) {
      char *model_name = strchr(buffer, ':');
      if (model_name != NULL) {
        printf("CPU Model: %s\n\n", model_name + 2); // Skip the colon and space
        break;
      }
    }
  }
  fclose(fp);
}

int main(void) {
  print_cpu_model();
  benchmark_executor<rwq>("moodycamel::ReaderWriterQueue");
  benchmark_executor<aqb>("max0x7ba/OptimistAtomicQueues");
  benchmark_executor<rq>("xenium/ramalhete_queue");
  benchmark_executor<dq>("std::deque with std::mutex");
  return 0;
}
