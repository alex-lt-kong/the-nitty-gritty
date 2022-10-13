#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
  srand(time(NULL));
  size_t dim[] = {100000, 1000000, 10000000, 100000000, 1000000000};
  struct timespec ts;
  uint32_t* arr_ptr;
  double delta, t0;

  
  int strides[] = {20391, 8433, 74508, 15065, 72462, 88331, 29235, 04731}; // 64 bytes cache line can store 4 unsigned int
  const size_t strides_count = sizeof(strides) / sizeof(strides[0]);
  for (int i = 0; i < sizeof(dim) / sizeof(dim[0]); ++i) {
    size_t d = dim[i];

    printf("%11lu,\t", d);

/*
    arr_ptr = (uint32_t*)malloc(d * sizeof(uint32_t));
    for (int j = 0; j < d; ++j) {
        arr_ptr[j] = j;
    }
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    for (size_t j = 0; j < d - 99999; ++j) {
      arr_ptr[j + strides[j % strides_count]] += arr_ptr[j-1];
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%12.9lf,\t%9u,\t", delta, arr_ptr[ts.tv_nsec % d]);
    free(arr_ptr);*/

    
    arr_ptr = (uint32_t*)malloc(d * sizeof(uint32_t));
    for (int j = 0; j < d; ++j) {
        arr_ptr[j] = j;
    }
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    for (size_t j = 0; j < d - 99999; ++j) {
      __builtin_prefetch(&arr_ptr[(j+1) + strides[(j+1) % strides_count]], 1, 1);
      arr_ptr[j + strides[j % strides_count]] += arr_ptr[j-1];
    //  printf("%lu, %lu\n", j + strides[j % strides_count], (j+4) + strides[(j+4) % strides_count]);
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%12.9lf,\t%9u,\t", delta, arr_ptr[ts.tv_nsec % d]);
    free(arr_ptr);

    printf("\n");
  }
  return 0;
}