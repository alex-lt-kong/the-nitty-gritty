#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define ITER 32
#define SIZE 33554432 // 32 * 1024 * 1024

struct pixelArray {
  float* r;
  float* g;
  float* b;
};

struct restrictPixelArray {
  float* restrict r;
  float* restrict g;
  float* restrict b;
};

void floating_division_potential_aliasing(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len);

void floating_division_ivdep(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len);

void floating_division_restrict(float a, float b, float c, struct restrictPixelArray* arr, struct restrictPixelArray* results, size_t arr_len);
