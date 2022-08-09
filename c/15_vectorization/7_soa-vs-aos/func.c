#include "func.h"

void floating_division_aos(float a, float b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results[i]->r = arr[i]->r / a;
    results[i]->g = b / arr[i]->g;
    results[i]->b = arr[i]->b / c;
  }
}

void floating_division_soa(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results->r[i] = arr->r[i] / a;
    results->g[i] = b / arr->g[i];
    results->b[i] = arr->b[i] / c;
  }
}


