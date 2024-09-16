#include "func.h"

void linear_func(uint8_t a, uint8_t b, uint8_t* arr, uint8_t* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}