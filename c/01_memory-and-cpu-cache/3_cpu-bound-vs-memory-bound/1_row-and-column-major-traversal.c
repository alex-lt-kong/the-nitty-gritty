#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  size_t dim[] = {10, 50, 100};
  struct timespec ts;
  double delta;
  for (int i = 0; i < sizeof(dim) / sizeof(dim[0]); ++i) {
    size_t d = dim[i];
    printf("dim: %lu, (%lu kb)\n", d, d * d / 1024);

    uint32_t arr1[d][d];
    timespec_get(&ts, TIME_UTC);
    double t0 = ts.tv_sec + (ts.tv_nsec / 1000) / 1000.0 / 1000.0;
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        arr1[j][k] += (j + k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + (ts.tv_nsec / 1000) / 1000.0 / 1000.0 - t0;
    printf("row-major:\t%lf, sample: %u\n", delta, arr1[ts.tv_sec % d * d][ts.tv_nsec % d]);

    uint32_t* arr1_ptr = (uint32_t*)malloc(dim[i] * dim[i] * 4);
    if (arr1_ptr == NULL) {
      perror("malloc()");
      return 1;
    }
    timespec_get(&ts, TIME_UTC);
    double t2 = ts.tv_sec + (ts.tv_nsec / 1000) / 1000.0 / 1000.0;
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        (*(arr1_ptr + (j*d) + k)) += (j + k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + (ts.tv_nsec / 1000) / 1000.0 / 1000.0 - t2;
    printf("row-major:\t%lf, sample: %u\n", delta, *(arr1_ptr + (ts.tv_sec % d * d) + ts.tv_nsec % d));
    free(arr1_ptr);

    uint32_t* arr2_ptr = (uint32_t*)malloc(dim[i] * dim[i] * 4);
    if (arr1_ptr == NULL) {
      perror("malloc()");
      return 1;
    }
    timespec_get(&ts, TIME_UTC);
    double t3 = ts.tv_sec + (ts.tv_nsec / 1000) / 1000.0 / 1000.0;
    for (int j = 0; j < dim[i]; ++j) {
      for (int k = 0; k < dim[i]; ++k) {
        *(arr2_ptr + (k*dim[i]) + j) += (j + k);
      }
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + (ts.tv_nsec / 1000) / 1000.0 / 1000.0 - t3;
    printf("column-major:\t%lf, sample: %u\n", delta, *(arr2_ptr + (ts.tv_sec % d * d) + ts.tv_nsec % d));
    free(arr2_ptr);
  }
  return 0;
}