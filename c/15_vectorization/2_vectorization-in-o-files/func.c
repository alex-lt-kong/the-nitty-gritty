#include "func.h"

void linear_func_external32(const uint32_t* a, const uint32_t* b, uint32_t* results, const size_t arr_len) {
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a[i] * b[i];
  }
}

void linear_func_external16(const uint16_t* a, const uint16_t* b, uint16_t* results, const size_t arr_len) {
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a[i] * b[i];
  }
}

void linear_func_external8(const uint8_t* a, const uint8_t* b, uint8_t* results, const size_t arr_len) {
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a[i] * b[i];
  }
}
