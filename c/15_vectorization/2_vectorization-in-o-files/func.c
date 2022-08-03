#include "func.h"


void linear_func_uint32(const uint32_t* arr, uint32_t* results, const size_t arr_len) {
  const unsigned int a = rand() % 1024;
  const unsigned int b = rand() % 1024;
  #pragma ivdep
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}

void linear_func_uint8(const uint8_t* arr, uint8_t* results, const size_t arr_len) {
  uint8_t a = rand() % 1024;
  uint8_t b = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}
