#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
  size_t dim[] = {10, 20, 50, 100, 150, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000};
  struct timespec ts;
  uint32_t* arr_ptr;
  double delta, t0;
  printf(
    "Dim,\tArraySize(KB),\tRow-Major Time,\tRM Sample,\tCol-Major Time,\tCM Sample\n"
  );
  for (int i = 0; i < sizeof(dim) / sizeof(dim[0]); ++i) {
    size_t d = dim[i];
    printf("%5lu,\t%11lu,\t", d, d * d * sizeof(uint32_t) / 1024);
    arr_ptr = (uint32_t*)malloc(d * d * sizeof(uint32_t));
    for (int j = 0; j < d; ++j) { // the initialized loop that all memory blocks are up and ready.
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
    printf("%0.9lf,\t%8u,\t", delta, *(arr_ptr + ts.tv_sec % d * d + ts.tv_nsec % d));
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
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%12.9lf,\t%9u\n", delta, *(arr_ptr + ts.tv_sec % d * d + ts.tv_nsec % d));
    free(arr_ptr);
  }
  return 0;
}