#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "../utils.h"
#include "func.h"

void linear_func_internal32(const uint32_t* a, const uint32_t* b, uint32_t* results, const size_t arr_len) {
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a[i] * b[i];
  }
}

void linear_func_internal16(const uint16_t* a, const uint16_t* b, uint16_t* results, const size_t arr_len) {
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a[i] * b[i];
  }
}

void linear_func_internal8(const uint8_t* a, const uint8_t* b, uint8_t* results, const size_t arr_len) {
  for (size_t i = 0; i < arr_len; ++i) {
    results[i] = a[i] * b[i];
  }
}

int main() {
  uint32_t* results32; 
  uint16_t* results16 = malloc(SIZE * sizeof(uint16_t));
  uint8_t*  results8 = malloc(SIZE * sizeof(uint8_t));

  uint32_t* a32;
  uint16_t* a16 = malloc(SIZE * sizeof(uint16_t));
  uint8_t*  a8 = malloc(SIZE * sizeof(uint8_t));
  uint32_t* b32;
  uint16_t* b16 = malloc(SIZE * sizeof(uint16_t));
  uint8_t*  b8 = malloc(SIZE * sizeof(uint8_t));

  uint64_t start_time;

  double* elapsed_times = malloc(ITER * sizeof(double));
  srand(time(NULL));



  printf("Calling linear_func_internal32()...\n");  
  for (int j = 0; j < ITER; ++j) {
    a32 = malloc(SIZE * sizeof(uint32_t));
    b32 = malloc(SIZE * sizeof(uint32_t));
    results32 = malloc(SIZE * sizeof(uint32_t));
    // We have to malloc() memory each time; otherwise CPU's caching could have a huge impact on the time needed.
    for (int i = 0; i < SIZE; ++i) {
      a32[i] = rand();
      b32[i] = rand();
    }

    start_time = get_timestamp_now();    
    linear_func_internal32(a32, b32, results32, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results32[rand() % SIZE]);
    // we pick and print one element from results, so that even the smartest compiler cant optimize the loop away.

    free(a32); free(b32); free(results32);
  }
  printf("\n");
  uint32_t avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %.2lf\n\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));
  printf("Calling linear_func_external32()...\n");  
  for (int j = 0; j < ITER; ++j) {
    a32 = malloc(SIZE * sizeof(uint32_t));
    b32 = malloc(SIZE * sizeof(uint32_t));
    results32 = malloc(SIZE * sizeof(uint32_t));
    for (int i = 0; i < SIZE; ++i) {
      a32[i] = rand();
      b32[i] = rand();
    }

    start_time = get_timestamp_now();
    linear_func_external32(a32, b32, results32, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results32[rand() % SIZE]);

    free(a32); free(b32); free(results32);
  }
  printf("\n");
  avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %.2lf\n\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));

  printf("Calling linear_func_internal16()...\n");  
  for (int j = 0; j < ITER; ++j) {
    a16 = malloc(SIZE * sizeof(uint16_t));
    b16 = malloc(SIZE * sizeof(uint16_t));
    results16 = malloc(SIZE * sizeof(uint16_t));
    for (int i = 0; i < SIZE; ++i) {
      a16[i] = rand();
      b16[i] = rand();
    }

    start_time = get_timestamp_now();
    linear_func_internal16(a16, b16, results16, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results16[rand() % SIZE]);

    free(a16); free(b16); free(results16);
  }
  printf("\n");
  avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %.2lf\n\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));
  printf("Calling linear_func_external16()...\n");  
  for (int j = 0; j < ITER; ++j) {
    a16 = malloc(SIZE * sizeof(uint16_t));
    b16 = malloc(SIZE * sizeof(uint16_t));
    results16 = malloc(SIZE * sizeof(uint16_t));
    for (int i = 0; i < SIZE; ++i) {
      a16[i] = rand();
      b16[i] = rand();
    }

    start_time = get_timestamp_now();
    linear_func_external16(a16, b16, results16, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results16[rand() % SIZE]);

    free(a16); free(b16); free(results16);
  }
  printf("\n");
  avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %.2lf\n\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));

  printf("Calling linear_func_internal8()...\n");  
  for (int j = 0; j < ITER; ++j) {
    a8 = malloc(SIZE * sizeof(uint16_t));
    b8 = malloc(SIZE * sizeof(uint16_t));
    results8 = malloc(SIZE * sizeof(uint16_t));
    for (int i = 0; i < SIZE; ++i) {
      a8[i] = rand();
      b8[i] = rand();
    }

    start_time = get_timestamp_now();
    linear_func_internal8(a8, b8, results8, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results8[rand() % SIZE]);
    free(a8); free(b8); free(results8);
  }
  printf("\n");
  avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %.2lf\n\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));
  printf("Calling linear_func_external8()...\n");  
  for (int j = 0; j < ITER; ++j) {    a8 = malloc(SIZE * sizeof(uint16_t));
    a8 = malloc(SIZE * sizeof(uint16_t));
    b8 = malloc(SIZE * sizeof(uint16_t));
    results8 = malloc(SIZE * sizeof(uint16_t));
    for (int i = 0; i < SIZE; ++i) {
      a8[i] = rand();
      b8[i] = rand();
    }

    start_time = get_timestamp_now();
    linear_func_external8(a8, b8, results8, SIZE);
    elapsed_times[j] = get_timestamp_now() - start_time;
    printf("%.0lfms(%u), ", elapsed_times[j], results8[rand() % SIZE]);
    free(a8); free(b8); free(results8);
  }
  printf("\n");
  avg_et = 0;
  for (int j = 0; j < ITER; ++j) {
      avg_et += elapsed_times[j];
  }
  printf("Average: %lums, std: %.2lf\n\n", avg_et / ITER, standard_deviation(elapsed_times, ITER, true));

  free(elapsed_times);
  return 0;
}