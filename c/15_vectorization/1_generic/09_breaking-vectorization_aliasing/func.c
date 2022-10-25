#include "func.h"


void floating_division_restrict(float a, float b, float c, struct restrictPixelArray* restrict arr, struct restrictPixelArray* restrict results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results->r[i] = arr->r[i] / a;
    results->g[i] = b / arr->g[i];
    results->b[i] = arr->b[i] / c;
  }
}

// ivdep means to ignore assumed vector dependencies
void floating_division_ivdep(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
  #if defined( __INTEL_COMPILER)
  #pragma ivdep
  #elif defined(__GNUC__)
  #pragma GCC ivdep
  #endif
  for (int i = 0; i < arr_len; ++i) {
    results->r[i] = arr->r[i] / a;
    results->g[i] = b / arr->g[i];
    results->b[i] = arr->b[i] / c;
  }
}


void floating_division_potential_aliasing(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    results->r[i] = arr->r[i] / a;
    results->g[i] = b / arr->g[i];
    results->b[i] = arr->b[i] / c;
  }
}

