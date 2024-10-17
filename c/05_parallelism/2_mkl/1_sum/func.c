#include "../../utils.h"

#include <mkl.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


double mkl_sum(double *arr, size_t arr_size) {
  // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-asum.html
  return cblas_dasum(arr_size, arr, 1);
  ;
}

double my_sum(double *arr, size_t arr_size) {
  double sum = 0;
  for (int j = 0; j < arr_size; ++j) {
    sum += arr[j];
  }
  return sum;
}

int main() {
  const size_t arr_size = 1000 * 1000 * 1000;
  float max_float = 1;
  double *arr = malloc(arr_size * sizeof(double));
  double sum1, sum2;
  uint64_t t0, t1;
  for (int j = 0; j < arr_size; ++j) {
    arr[j] = (double)rand() / (double)(RAND_MAX / max_float);
  }
  t0 = get_timestamp_in_microsec();
  sum1 = mkl_sum(arr, arr_size);
  t1 = get_timestamp_in_microsec();
  printf("mkl_sum(): %lf, takes %lf ms\n", sum1, (t1 - t0) / 1000.0);
  t0 = get_timestamp_in_microsec();
  sum2 = my_sum(arr, arr_size);
  t1 = get_timestamp_in_microsec();
  printf("my_sum(): %lf, takes %lf ms\n", sum2, (t1 - t0) / 1000.0);
  return 0;
}