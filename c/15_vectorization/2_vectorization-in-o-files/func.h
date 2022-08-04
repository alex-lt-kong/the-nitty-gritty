#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define SIZE 67108864 // 64 * 1024 * 1024
#define ITER 64

void linear_func_external32(const uint32_t* a, const uint32_t* b, uint32_t* results, const size_t arr_len);

void linear_func_external16(const uint16_t* a, const uint16_t* b, uint16_t* results, const size_t arr_len);

void linear_func_external8(const uint8_t* a, const uint8_t* b, uint8_t* results, const size_t arr_len);
