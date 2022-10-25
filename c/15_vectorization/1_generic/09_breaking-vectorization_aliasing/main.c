#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>
#include <time.h>

#include "func.h"

void jobs_pixelarray(void (*fun_ptr)(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len)) {
  double elapsed_times[ITER];
  struct pixelArray* arr = malloc(sizeof(struct pixelArray*));
  struct pixelArray* results = malloc(sizeof(struct pixelArray*));
  float a = (rand() % INT32_MAX) / 3.14;
  float b = (rand() % INT32_MAX) / 2.71;
  float c = (rand() % INT32_MAX) / 1.73;
  
  arr->r = malloc(SIZE * sizeof(float));
  arr->g = malloc(SIZE * sizeof(float));
  arr->b = malloc(SIZE * sizeof(float));
  results->r = malloc(SIZE * sizeof(float));
  results->g = malloc(SIZE * sizeof(float));
  results->b = malloc(SIZE * sizeof(float));
  for (uint64_t i = 0; i < ITER; ++i) {

    for (int j = 0; j < SIZE; ++j) {
      arr->r[j] = (rand() % INT32_MAX) / 1.41;
      arr->g[j] = (rand() % INT32_MAX) / 1.41;
      arr->b[j] = (rand() % INT32_MAX) / 1.41;
    }
    clock_t start_time = clock();
    fun_ptr(a, b, c, arr, results, SIZE);
    elapsed_times[i] = clock() - start_time;
    if (i == 0 || rand() % 2 == 0) {
      printf(
        "%0.2lfms(%.02f,%.03f,%.02f), ", elapsed_times[i] / 1000,
        results->r[rand() % SIZE], results->g[rand() % SIZE], results->b[rand() % SIZE]
      );
    }
  }  
  printf("\n");
  double avg_et = 0;
  for (int i = 0; i < ITER; ++i) {
    avg_et += elapsed_times[i];
  }
  avg_et /= ITER;
  printf("avg: %0.2lfms\n", avg_et / 1000);
  
  free(arr);
  free(results);
}

void jobs_restrictpixelarray(void (*fun_ptr)(float a, float b, float c, struct restrictPixelArray* arr, struct restrictPixelArray* results, size_t arr_len)) {
  double elapsed_times[ITER];
  struct restrictPixelArray* arr = malloc(sizeof(struct restrictPixelArray*));
  struct restrictPixelArray* results = malloc(sizeof(struct restrictPixelArray*));
  float a = (rand() % INT32_MAX) / 3.14;
  float b = (rand() % INT32_MAX) / 2.71;
  float c = (rand() % INT32_MAX) / 1.73;
  
  arr->r = malloc(SIZE * sizeof(float));
  arr->g = malloc(SIZE * sizeof(float));
  arr->b = malloc(SIZE * sizeof(float));
  results->r = malloc(SIZE * sizeof(float));
  results->g = malloc(SIZE * sizeof(float));
  results->b = malloc(SIZE * sizeof(float));
  for (uint64_t i = 0; i < ITER; ++i) {

    for (int j = 0; j < SIZE; ++j) {
      arr->r[j] = (rand() % INT32_MAX) / 1.41;
      arr->g[j] = (rand() % INT32_MAX) / 1.41;
      arr->b[j] = (rand() % INT32_MAX) / 1.41;
    }
    clock_t start_time = clock();
    fun_ptr(a, b, c, arr, results, SIZE);
    elapsed_times[i] = clock() - start_time;
    if (i == 0 || rand() % 2 == 0) {
      printf(
        "%0.2lfms(%.02f,%.03f,%.02f), ", elapsed_times[i] / 1000,
        results->r[rand() % SIZE], results->g[rand() % SIZE], results->b[rand() % SIZE]
      );
    }
  }  
  printf("\n");
  double avg_et = 0;
  for (int i = 0; i < ITER; ++i) {
    avg_et += elapsed_times[i];
  }
  avg_et /= ITER;
  printf("avg: %0.2lfms\n", avg_et / 1000);
  
  free(arr);
  free(results);
}

int main() {
  srand(time(NULL));
  jobs_pixelarray(floating_division_potential_aliasing);
  jobs_pixelarray(floating_division_ivdep);  
  jobs_restrictpixelarray(floating_division_restrict);  
  return 0;
}