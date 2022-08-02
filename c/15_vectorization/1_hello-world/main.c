#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "../utils.h"

#define SIZE 134217728 // 1238 * 1024 * 1024

void linear_func(unsigned int* arr, unsigned int* results, size_t arr_len) {
  unsigned int a = rand() % 1024;
  unsigned int b = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}

int main() {
  const unsigned int ITER = 128;
  unsigned int* arr = malloc(SIZE * sizeof(unsigned int));
  unsigned int* results = calloc(SIZE, sizeof(unsigned int));
  unsigned long long* elapsed_times = (unsigned long long*)calloc(ITER, sizeof(unsigned long long));
  srand(time(NULL));
  struct timeval tv;
  for (int i = 0; i < SIZE; ++i) {
    arr[i] = rand() % SIZE;
  }
  
  for (int j = 0; j < ITER; ++j) {
    gettimeofday(&tv, NULL);
    unsigned long long start_time = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
    linear_func(arr, results, SIZE);
    gettimeofday(&tv, NULL);
    unsigned long long end_time = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
    elapsed_times[j] = end_time - start_time;
    printf("%llums(%u), ", elapsed_times[j], results[rand() % SIZE]);
    // we pick and print one element from results, so that even the smartest compiler cant optimize my loop away.
  }
  printf("\n");
  unsigned int avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %lf\n", avg_et / ITER, standard_deviation(elapsed_times, ITER));
  
  free(arr);
  free(results);
  free(elapsed_times);
  return 0;
}