#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
  srand(time(NULL));
  size_t dim[] = {
    5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 2000, 5000, 10000, 20000
  };
  struct timespec ts;
  uint32_t* arr_ptr;
  double delta, t0;
  printf(
    "Dim,\tArraySize(KB),\tRow-Major Time,\tSample,\t\tCol-Major Time,\tSample,\tFast Random Time,\tSample,\tProper Random Time,\tSample\n"
  );
  for (int i = 0; i < sizeof(dim) / sizeof(dim[0]); ++i) {
    size_t d = dim[i];
    size_t dd = d * d;

    printf("%5lu,\t%11lu,\t", d, dd * sizeof(uint32_t) / 1024);


    arr_ptr = (uint32_t*)malloc(dd * sizeof(uint32_t));
    for (int j = 0; j < d; ++j) { // the initialization loop which makes sure all memory blocks are up and ready.
      for (int k = 0; k < d; ++k) {
        *(arr_ptr + j * d + k) = (j * k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        *(arr_ptr + j * d + k) += (j + k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%0.9lf,\t%8u,\t", delta, *(arr_ptr + ts.tv_sec % dd + ts.tv_nsec % d));
    free(arr_ptr);


    arr_ptr = (uint32_t*)malloc(dd * sizeof(uint32_t));
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        *(arr_ptr + j * d + k) = (j * k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        *(arr_ptr + k * d + j) += (j + k);        
      }
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%12.9lf,\t%9u,\t", delta, *(arr_ptr + ts.tv_sec % dd + ts.tv_nsec % d));
    free(arr_ptr);


    arr_ptr = (uint32_t*)malloc(dd * sizeof(uint32_t));
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        *(arr_ptr + j * d + k) = (j * k);
      }
    }
    int seed = rand();
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    for (size_t j = 0; j < d; ++j) {
      for (size_t k = 0; k < d; ++k) {
        *(arr_ptr + (seed + d * j * k) % dd) += (j + k);
        // This so-called "fast random" is not really "random"--it is designed to confuse prefetcher only.
      }
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%12.9lf,\t%9u,\t", delta, *(arr_ptr + ts.tv_sec % dd + ts.tv_nsec % d));
    free(arr_ptr);


    arr_ptr = (uint32_t*)malloc(dd * sizeof(uint32_t));
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        *(arr_ptr + j * d + k) = (j * k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        *(arr_ptr + rand() % dd) += (j + k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%12.9lf,\t%9u\n", delta, *(arr_ptr + ts.tv_sec % dd + ts.tv_nsec % d));
    free(arr_ptr);
  }
  return 0;
}