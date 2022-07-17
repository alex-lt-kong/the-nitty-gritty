#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


void add_one_func(int* arr, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    arr[i] += 1;
  }
}

__attribute__((optimize("no-tree-vectorize")))
void add_one_no_vectorization(int* arr, size_t arr_len) {
  add_one_func(arr, arr_len);
}

void add_one_auto_vectorization(int* arr, size_t arr_len) {
  add_one_func(arr, arr_len);
}

int main() {
  const size_t SIZE = 64 * 1024 * 1024;
  const unsigned int ITER = 128;
  int* arr = calloc(SIZE, sizeof(int));
  unsigned long long* results = (unsigned long long*)calloc(ITER, sizeof(unsigned long long));
  srand(time(NULL));
  struct timeval tv;
  for (int i = 0; i < SIZE; ++i) {
    arr[i] = rand() % SIZE;
  }

  void (*funcs[2])() = {
    add_one_auto_vectorization,
    add_one_no_vectorization
  };
  char func_names[][64] = {
    "add_one_auto_vectorization",
    "add_one_no_vectorization"
  };
  for (int i = 0; i < 2; ++i) {
    printf("%s:\n", func_names[i]);
    for (int j = 0; j < ITER; ++j) {
      gettimeofday(&tv, NULL);
      unsigned long long start_time = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
      (*funcs[i])(arr, SIZE);
      gettimeofday(&tv, NULL);
      unsigned long long end_time = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
      results[j] = end_time - start_time;
      printf("%llu ms,", results[j]);
      if ((j + 1) % 16 == 0 && j + 1 < ITER) { printf("\n"); }
    }
    printf("\n");
    unsigned int avg_result = 0;
    for (int j = 0; j < ITER; ++j) {
      avg_result += results[j];
    }
    printf("Average: %lu\n\n", avg_result / ITER);
  }
  free(arr);
  free(results);
  return 0;
}