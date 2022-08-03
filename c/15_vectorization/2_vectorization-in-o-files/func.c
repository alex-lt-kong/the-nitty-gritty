#include "func.h"


inline void linear_func(const unsigned int arr[], unsigned int results[], const size_t arr_len) {
  const unsigned int a = rand() % 1024;
  const unsigned int b = rand() % 1024;
  #pragma ivdep
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a * arr[i] + b;
  }
}
