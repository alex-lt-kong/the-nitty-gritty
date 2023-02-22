#include "func.h"

void linear_func_float(float a, float b, float* arr, float* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}

void linear_func_double(double a, double b, double* arr, double* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}