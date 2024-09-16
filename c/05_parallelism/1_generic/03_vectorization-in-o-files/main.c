#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "../utils.h"
#include "func.h"

#define SIZE 256


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

  uint64_t start_time = get_timestamp_in_microsec();
  for (uint64_t i = 0; i < 1024 * 1024 * 32; ++i) {
    linear_func(a, b, arr, results, SIZE);
    if (i % (1024 * 1024) == 0) {
      printf("%u\n", results[rand() % SIZE]);
    }
  }
  elapsed_times = get_timestamp_in_microsec() - start_time;

  free(arr);
  free(results);
  printf("%.2lfms\n", elapsed_times / 1000);
  return 0;
}