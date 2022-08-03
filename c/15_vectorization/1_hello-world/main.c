#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../utils.h"

#define SIZE 134217728 // 1238 * 1024 * 1024
#define ITER 128

void linear_func(unsigned int* arr, unsigned int* results, size_t arr_len) {
  unsigned int a = rand() % 1024;
  unsigned int b = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}

int main() {
  unsigned int* arr = malloc(SIZE * sizeof(unsigned int));
  unsigned int* results = malloc(SIZE * sizeof(unsigned int));
  double* elapsed_times = malloc(ITER * sizeof(double));
  srand(time(NULL));
  for (int i = 0; i < SIZE; ++i) {
    arr[i] = rand() % SIZE;
  }
  
  for (int j = 0; j < ITER; ++j) {
    unsigned long long start_time = get_timestamp_now();
    linear_func(arr, results, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%llums(%u), ", elapsed_times[j], results[rand() % SIZE]);
    // we pick and print one element from results, so that even the smartest compiler cant optimize my loop away.
  }
  printf("\n");
  unsigned int avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %lf\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));
  
  free(arr);
  free(results);
  free(elapsed_times);
  return 0;
}