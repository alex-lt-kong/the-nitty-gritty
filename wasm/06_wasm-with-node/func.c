#include <stdlib.h>
#include <stdint.h>
#include "emscripten.h"

EMSCRIPTEN_KEEPALIVE
int add(int a, int b) {
  return a + b;
}

EMSCRIPTEN_KEEPALIVE
int* my_Malloc(size_t size) {
  return malloc(size);
}

EMSCRIPTEN_KEEPALIVE
void my_free(int* ptr) {
  free(ptr);
}

EMSCRIPTEN_KEEPALIVE
int* bubble_sort(int* arr, int arr_len) {
  // Even if we sort the array in-place, we still need to return it given how the function is called from
  // the JavaScript side.
  for (int i = 0; i < arr_len; ++i) {
    for (int j = i; j < arr_len; ++j) {
      if (arr[i] < arr[j]) {
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
    }
  }
  return arr;
}