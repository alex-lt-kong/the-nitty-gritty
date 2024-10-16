#include <mkl.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

uint64_t get_timestamp_in_microsec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1000000 * tv.tv_sec + tv.tv_usec;
}

double mkl_multiplication(double *arr, double multiplier, size_t arr_size) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();
  // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-scal.html
  cblas_dscal(arr_size, multiplier, arr, 1);
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}

double my_multiplication(double *arr, double multiplier, size_t arr_size) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();
  for (int j = 0; j < arr_size; ++j) {
    arr[j] *= multiplier;
  }
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}
