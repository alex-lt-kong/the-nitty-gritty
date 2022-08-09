#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "../utils.h"
#include "func.h"

int main() {
  srand(time(NULL));
  double elapsed_times[ITER];
  struct pixel** arr = malloc(SIZE * sizeof(struct pixel*));
  struct pixel** results = malloc(SIZE * sizeof(struct pixel*));
  float a = (rand() % INT32_MAX) / 3.14;
  float b = (rand() % INT32_MAX) / 2.71;
  float c = (rand() % INT32_MAX) / 1.73;
  
  for (uint64_t i = 0; i < ITER; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      arr[j] = malloc(sizeof(struct pixel));
      results[j] = malloc(sizeof(struct pixel));
      arr[j]->r = (rand() % INT32_MAX) / 1.41;
      arr[j]->g = (rand() % INT32_MAX) / 1.41;
      arr[j]->b = (rand() % INT32_MAX) / 1.41;
    }
    clock_t start_time = clock();
    floating_division_aos(a, b, c, arr, results, SIZE);
    elapsed_times[i] = clock() - start_time;
    if (i == 0 || rand() % 2 == 0) {
      printf(
        "%0.2lfms(%.02f,%.03f,%.02f), ", elapsed_times[i] / 1000,
        results[rand() % SIZE]->r, results[rand() % SIZE]->g, results[rand() % SIZE]->b
      );
    }    
    for (int i = 0; i < SIZE; ++i) {
      free(arr[i]);
      free(results[i]);
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