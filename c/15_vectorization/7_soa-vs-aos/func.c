#include "func.h"

/*
From: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/optimization-and-programming/vectorization/automatic-vectorization/using-automatic-vectorization.html

#pragma ivdep: may be used to tell the compiler that it may safely ignore any potential data dependencies.
(The compiler will not ignore proven dependencies). Use of this pragma when there are dependencies may lead
to incorrect results.
There are cases where the compiler cannot tell by a static dependency analysis that it is safe to vectorize.
Consider the following loop: 

void copy(char *cp_a, char *cp_b, int n) {
  for (int i = 0; i < n; i++) { cp_a[i] = cp_b[i]; } 
}

Without more information, a vectorizing compiler must conservatively assume that the memory regions accessed by the
pointer variables cp_a and cp_b may (partially) overlap, which gives rise to potential data dependencies that
prohibit straightforward conversion of this loop into SIMD instructions. 
*/

void floating_division_aos(float a, float b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
  
  #ifdef __INTEL_COMPILER
  #pragma ivdep
  // Pragmas are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.
  // https://stackoverflow.com/questions/5078679/what-is-the-scope-of-a-pragma-directive
  #endif
  for (int i = 0; i < arr_len; ++i) {
    results[i]->r = arr[i]->r / a;
    results[i]->g = b / arr[i]->g;
    results[i]->b = arr[i]->b / c;
  }
}

void floating_division_soa(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
  #ifdef __INTEL_COMPILER
  #pragma ivdep  
  #endif
  for (int i = 0; i < arr_len; ++i) {
    results->r[i] = arr->r[i] / a;
    results->g[i] = b / arr->g[i];
    results->b[i] = arr->b[i] / c;
  }
}


