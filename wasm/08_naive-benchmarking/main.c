#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "func.c"

#define ITER 16

int main(void) {
  const size_t SIZE = 1024 * 1024 * 2;   
  srand(time(NULL));
  int* in_array = calloc(SIZE, sizeof(int));
  int results[ITER] = {0};
  struct timeval tv;
  for (int i = 0; i < ITER; ++i) {
    printf("%d-it iteration...\n", i);
    for(int j = 0; j < SIZE; ++j) {
      in_array[j] = rand();  // RAND_MAX should be 2147483647
    }
       
    gettimeofday(&tv, NULL);
    unsigned long long start_time = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;

    quick_sort(in_array, 0, SIZE-1);

    gettimeofday(&tv, NULL);
    results[i] = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000 - start_time;
    
  }
  free(in_array);
  printf("quickSort in C: ");
  int avg = 0;
  for (int i = 0; i < ITER; ++i) {
    avg += results[i];
    printf("%d,", results[i]);
  }
  printf(" avg: %d\n", avg / ITER);
  return 0;
} 