#include <stdio.h>
#include <time.h>

#include "../utils.h"
#include "func.h"

#define SIZE 134217728 // 128 * 1024 * 1024
#define ITER 128

int main() {
  uint32_t* arr32 = malloc(SIZE * sizeof(uint32_t));
  uint32_t* results32 = malloc(SIZE * sizeof(uint32_t));
  uint8_t* arr8 = malloc(SIZE * sizeof(uint8_t));
  uint8_t* results8 = malloc(SIZE * sizeof(uint8_t));
  double* elapsed_times = malloc(ITER * sizeof(double));
  srand(time(NULL));
  for (int i = 0; i < SIZE; ++i) {
    arr32[i] = rand() % SIZE;
  }
  
  for (int j = 0; j < ITER; ++j) {
    unsigned long long start_time = get_timestamp_now();
    linear_func_uint32(arr32, results32, SIZE);
    //linear_func_uint8(arr8, results8, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results32[rand() % SIZE]);
    //printf("%.0lfms(%u), ", elapsed_times[j], results8[rand() % SIZE]);
    // we pick and print one element from results, so that even the smartest compiler cant optimize my loop away.
  }
  printf("\n");
  unsigned int avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %lf\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));
  
  free(arr32);
  free(results32);
  free(arr8);
  free(results8);
  free(elapsed_times);
  return 0;
}