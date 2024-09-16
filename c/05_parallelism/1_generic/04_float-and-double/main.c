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
  float* arr_f = malloc(SIZE * sizeof(float));
  float* results_f = malloc(SIZE * sizeof(float));
  float a_f = (rand() % 1024) * 1.0;
  float b_f = (rand() % 1024) * 1.0;
  for (int i = 0; i < SIZE; ++i) {
    arr_f[i] = (rand() % SIZE) * 1.414;
  }

  uint64_t start_time = get_timestamp_in_microsec();
  for (uint64_t i = 0; i < 1024 * 1024 * 32; ++i) {
    linear_func_float(a_f, b_f, arr_f, results_f, SIZE);
    if (i % (1024 * 1024 * 2) == 0) {
      printf("%f\n", results_f[rand() % SIZE]);
    }
  }
  elapsed_times = get_timestamp_in_microsec() - start_time;

  free(arr_f);
  free(results_f);
  printf("%.2lfms\n\n", elapsed_times / 1000);

  double* arr_d = malloc(SIZE * sizeof(double));
  double* results_d = malloc(SIZE * sizeof(double));
  double a_d = (rand() % 1024) * 2.71;
  double b_d = (rand() % 1024) * 3.14;
  for (int i = 0; i < SIZE; ++i) {
    arr_d[i] = (rand() % SIZE) * 1.414;
  }

  start_time = get_timestamp_in_microsec();
  for (uint64_t i = 0; i < 1024 * 1024 * 32; ++i) {
    linear_func_double(a_d, b_d, arr_d, results_d, SIZE);
    if (i % (1024 * 1024 * 2) == 0) {
      printf("%.4lf\n", results_d[rand() % SIZE]);
    }
  }
  elapsed_times = get_timestamp_in_microsec() - start_time;

  free(arr_d);
  free(results_d);
  printf("%.2lfms\n", elapsed_times / 1000);
  return 0;
}