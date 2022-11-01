#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
  srand(time(NULL));
  
  struct timespec ts;
  uint8_t sum = 0;
  double delta, t0;
  size_t arr_len, idx;
  const size_t repeats = 64 * 1024 * 1024;
  uint32_t* arr;
  for (size_t step = 1; step < 1024; ++step) {
    arr_len = step * 1024 * 1024 / sizeof(uint32_t);
    arr = malloc(sizeof(uint32_t) * arr_len);
    for (size_t i = 0; i < arr_len; ++i) {
        arr[i] = rand();
    }
    timespec_get(&ts, TIME_UTC);
    t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
    idx = 0;

    for (size_t i = 0; i < repeats; ++i) {        
        arr[idx] += step;
        sum += arr[idx];
        idx = (idx + step) % arr_len;
    }
    timespec_get(&ts, TIME_UTC);
    delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
    printf("%lu, %lf\n", step, delta);
    free(arr);
  }
  return sum;
}