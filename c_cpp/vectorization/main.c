#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// unsigned int never overflows, which is exactly what we need
void add_one_func(unsigned int* arr, size_t arr_len) {
  for (int i = 0; i < arr_len; ++i) {
    arr[i] += 1;
  }
}

void linear_func(unsigned int* arr, size_t arr_len) {
  unsigned int a = rand() % 1024;
  unsigned int b = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    arr[i] = a * arr[i] + b;
  }
}

void quadratic_func(unsigned int* arr, size_t arr_len) {
  unsigned int a = rand() % 1024;
  unsigned int b = rand() % 1024;
  unsigned int c = rand() % 1024;
  for (int i = 0; i < arr_len; ++i) {
    arr[i] = a * arr[i] * arr[i] + b * arr[i] + c;
  }
}

__attribute__((optimize("no-tree-vectorize")))
void add_one_vectorization_off(unsigned int* arr, size_t arr_len) {
  add_one_func(arr, arr_len);
}

void add_one_vectorization_on(unsigned int* arr, size_t arr_len) {
  add_one_func(arr, arr_len);
}

__attribute__((optimize("no-tree-vectorize")))
void linear_vectorization_off(unsigned int* arr, size_t arr_len) {
  linear_func(arr, arr_len);
}

void linear_vectorization_on(unsigned int* arr, size_t arr_len) {
  linear_func(arr, arr_len);
}

__attribute__((optimize("no-tree-vectorize")))
void quadratic_vectorization_off(unsigned int* arr, size_t arr_len) {
  quadratic_func(arr, arr_len);
}

void quadratic_vectorization_on(unsigned int* arr, size_t arr_len) {
  quadratic_func(arr, arr_len);
}

int main() {
  const size_t SIZE = 64 * 1024 * 1024;
  const unsigned int ITER = 128;
  unsigned int* arr = calloc(SIZE, sizeof(unsigned int));
  unsigned long long* results = (unsigned long long*)calloc(ITER, sizeof(unsigned long long));
  srand(time(NULL));
  struct timeval tv;
  for (int i = 0; i < SIZE; ++i) {
    arr[i] = rand() % SIZE;
  }

  void (*funcs[])() = {
    add_one_vectorization_off,
    add_one_vectorization_on,
    linear_vectorization_off,
    linear_vectorization_on,
    quadratic_vectorization_off,
    quadratic_vectorization_on
  };
  char func_names[][64] = {
    "add_one_vectorization_off",
    "add_one_vectorization_on",
    "linear_vectorization_off",
    "linear_vectorization_on",
    "quadratic_vectorization_off",
    "quadratic_vectorization_on"
  };
  for (int i = 0; i < sizeof(funcs)/sizeof(funcs[0]); ++i) {
    printf("%s:\n", func_names[i]);
    for (int j = 0; j < ITER; ++j) {
      gettimeofday(&tv, NULL);
      unsigned long long start_time = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
      (*funcs[i])(arr, SIZE);
      gettimeofday(&tv, NULL);
      unsigned long long end_time = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
      results[j] = end_time - start_time;
      printf("%llums,", results[j]);
      if ((j + 1) % 32 == 0 && j + 1 < ITER) { printf("\n"); }
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