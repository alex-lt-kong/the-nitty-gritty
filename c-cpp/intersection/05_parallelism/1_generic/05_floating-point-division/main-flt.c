#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "../utils.h"
#include "func.h"

#define ITER 256
#define SIZE 33554432 // 32 * 1024 * 1024


int main() {
  srand(time(NULL));
  double elapsed_times[ITER];
  float* arr = malloc(SIZE * sizeof(float));
  float* results = malloc(SIZE * sizeof(float));
  float a = (rand() % INT32_MAX) / 3.14;
  float b = (rand() % INT32_MAX) / 2.71;
  float c = (rand() % INT32_MAX) / 1.73;
  float d = (rand() % INT32_MAX) / 1.61;
  
  for (uint64_t i = 0; i < ITER; ++i) {    
    for (int j = 0; j < SIZE; ++j) {
      arr[j] = (rand() % INT32_MAX) / 1.41;
    }
    uint64_t start_time = get_timestamp_in_microsec();
    func_floating_division(a, b, c, d, arr, results, SIZE);
    elapsed_times[i] = get_timestamp_in_microsec() - start_time;
    if (i == 0 || rand() % 4 == 0) {
      printf("%0.2lfms(%.02f), ", elapsed_times[i] / 1000, results[rand() % SIZE]);
    }    
  }  
  printf("\n");
  double avg_et = 0;
  for (int i = 0; i < ITER; ++i) {
    avg_et += elapsed_times[i];
  }
  avg_et /= ITER;
  printf("avg: %0.2lfms, std: %0.2lf\n", avg_et / 1000, standard_deviation(elapsed_times, ITER, true));
  
  free(arr);
  free(results);
  return 0;
}