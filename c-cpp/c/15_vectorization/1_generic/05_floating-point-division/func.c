#include "func.h"

void func_floating_division(float a, float b, float c, float d, float* arr, float* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i] =  arr[i] / a;
    results[i] += b / arr[i];
    results[i] += arr[i] / c;
    results[i] += d / arr[i];
  }
}

void func_int_multiplication(int32_t a, int32_t b, int32_t c, int32_t d, int32_t* arr, int32_t* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i] =  arr[i] * a;
    results[i] += arr[i] * b;
    results[i] += arr[i] * c;
    results[i] += arr[i] * d;
  }
}