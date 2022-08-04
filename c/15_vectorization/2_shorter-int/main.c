#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "../utils.h"

#define SIZE 256
// The idea is that, the SIZE must be small enough to fit into CPU cache, so that we can test the "real" performance
// of CPU, instead of keeping it waiting for data to be fetched from memory.

void linear_func(uint8_t a, uint8_t b, uint8_t* arr, uint8_t* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}

int main() {
  srand(time(NULL));
  double elapsed_times;
  uint8_t* arr = malloc(SIZE * sizeof(uint8_t));
  uint8_t* results = malloc(SIZE * sizeof(uint8_t));
  uint8_t a = rand() % 1024;
  uint8_t b = rand() % 1024;
  for (int j = 0; j < SIZE; ++j) {
    arr[j] = rand() % SIZE;
  }

  uint64_t start_time = get_timestamp_now();
  for (uint64_t i = 0; i < 1024 * 1024 * 32; ++i) {
    linear_func(a, b, arr, results, SIZE);
    if (i % (1024 * 1024) == 0) {
      printf("%u\n", results[rand() % SIZE]);
    }
  }
  elapsed_times = get_timestamp_now() - start_time;

  free(arr);
  free(results);
  printf("%.2lfms\n", elapsed_times / 1000);
  return 0;
}