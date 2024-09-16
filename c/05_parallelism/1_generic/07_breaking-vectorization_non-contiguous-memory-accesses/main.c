#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define ARR_SIZE 65536

int main() {

  srand(time(NULL));
  struct timespec ts;
  double delta, t0;

  uint8_t* a = malloc(sizeof(uint8_t) * ARR_SIZE);
  uint8_t* b = malloc(sizeof(uint8_t) * ARR_SIZE);
  uint8_t* c1 = malloc(sizeof(uint8_t) * ARR_SIZE);
  uint8_t* c2 = malloc(sizeof(uint8_t) * ARR_SIZE);

  for (int i = 0; i < ARR_SIZE; ++i) {
    a[i] = rand();
    b[i] = rand();
    c1[i] = 0;
    c2[i] = 0;
  }

  timespec_get(&ts, TIME_UTC);
  t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
  for (int i = 0; i < ARR_SIZE; ++i) {
    c1[i] += a[i] * b[i];
  }
  timespec_get(&ts, TIME_UTC);
  delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
  printf("stride: 1,\t%0.9lf,\t%8u\n", delta, *(c1 + ts.tv_sec % ARR_SIZE));

  timespec_get(&ts, TIME_UTC);
  t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
  
  const int stride = 4;
  for (int i = 0; i < ARR_SIZE; i += stride) {
    c2[i] += a[i] * b[i];
  }
  timespec_get(&ts, TIME_UTC);
  delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
  printf("stride: %d,\t%0.9lf,\t%8u\n", stride, delta, *(c2 + ts.tv_sec % ARR_SIZE));

  free(a);
  free(b);
  free(c1);
  free(c2);
  return 0;
}