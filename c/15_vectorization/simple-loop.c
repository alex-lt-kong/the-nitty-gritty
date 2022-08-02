#include "simple-loop.h"

// unsigned int never overflows, which is exactly what we need
void add_one_func(unsigned int* arr, unsigned int* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i] = arr[i] + 1;
  }
}

void linear_func(unsigned int* arr, unsigned int* results, size_t arr_len) {
  unsigned int a = rand() % 1024;
  unsigned int b = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}

void quadratic_func(unsigned int* arr, unsigned int* results, size_t arr_len) {
  unsigned int a = rand() % 1024;
  unsigned int b = rand() % 1024;
  unsigned int c = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] * arr[i] + b * arr[i] + c;
  }
}
