#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define ITER 32
#define SIZE 33554432 // 32 * 1024 * 1024

struct pixel {
  float r;
  float g;
  float b;
};

struct pixelArray {
  float* r;
  float* g;
  float* b;
};

void floating_division_aos(float a, float b, float c, struct pixel** arr, struct pixel** results, size_t arr_len);

void floating_division_soa(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len);


