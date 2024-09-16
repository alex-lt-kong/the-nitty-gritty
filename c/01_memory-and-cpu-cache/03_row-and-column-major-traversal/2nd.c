#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
  size_t dim[] = {
    5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 2000, 5000, 10000, 20000, 30000
  };
  struct timespec ts;
  uint32_t* arr_ptr;
  double delta0, delta1, t0;
  printf(
    "Dim,\tArraySize(KB),\tRow-Major Time,\tRM Sample,\tCol-Major Time,\tCM Sample,\tDiff\n"
  );
  for (int i = 0; i < sizeof(dim) / sizeof(dim[0]); ++i) {
    size_t d = dim[i];
    printf("%5lu,\t%11lu,\t", d, d * d * sizeof(uint32_t) / 1024);
    arr_ptr = (uint32_t*)malloc(d * d * sizeof(uint32_t));
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
    delta0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%0.9lf,\t%8u,\t", delta0, *(arr_ptr + ts.tv_sec % d * d + ts.tv_nsec % d));
    free(arr_ptr);

    arr_ptr = (uint32_t*)malloc(d * d * sizeof(uint32_t));
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
    delta1 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%12.9lf,\t%9u,\t%0.9lf\n", delta1, *(arr_ptr + ts.tv_sec % d * d + ts.tv_nsec % d), delta1 - delta0);
    free(arr_ptr);
  }
  return 0;
}