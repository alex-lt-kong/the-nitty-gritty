#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
  srand(time(NULL));
  const uint64_t arr_len = 128 * 1024 * 1024 + 1;
  uint32_t* arr = malloc(sizeof(uint32_t) * arr_len);

  for (uint64_t i = 0; i < arr_len; ++i) {
    arr[i] = rand();
  }
  struct timespec ts;
  uint8_t sum = 0;
  double delta, t0;
  uint64_t idx;
  for (uint64_t step = 1; step < (8192) + 1; ++step) {
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    idx = 0;
    for (uint64_t i = 0; i < arr_len; ++i) {        
        arr[idx] += step;
        sum += arr[idx];
        idx = (idx + step) % arr_len;
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%lu, %lf\n", step, delta);
  }

  free(arr);
  return sum;
}