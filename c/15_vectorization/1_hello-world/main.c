#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "../utils.h"

#define SIZE 134217728 // 1238 * 1024 * 1024
#define ITER 128

void linear_func(uint32_t* arr, uint32_t* results, size_t arr_len) {
  uint32_t a = rand() % 1024;
  uint32_t b = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}

int main() {
  uint32_t* arr = malloc(SIZE * sizeof(uint32_t));
  uint32_t* results = malloc(SIZE * sizeof(uint32_t));
  double* elapsed_times = malloc(ITER * sizeof(double));
  srand(time(NULL));
  for (int i = 0; i < SIZE; ++i) {
    arr[i] = rand() % SIZE;
  }
  
  for (int j = 0; j < ITER; ++j) {
    unsigned long long start_time = get_timestamp_now();
    linear_func(arr, results, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results[rand() % SIZE]);
    // we pick and print one element from results, so that even the smartest compiler cant optimize my loop away.
  }
  printf("\n");
  uint32_t avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %lf\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));
  
  free(arr);
  free(results);
  free(elapsed_times);
  return 0;
}