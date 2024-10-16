#include <mkl.h>

#include <math.h>
#include <stdint.h>
#include <sys/time.h>

uint64_t get_timestamp_in_microsec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1000000 * tv.tv_sec + tv.tv_usec;
}

double mkl_pow(uint64_t arr_size, double *restrict vec_base,
               double *restrict vec_exp, double *restrict vec_out) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();
  // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/vector-mathematical-functions/vm-mathematical-functions/power-and-root-functions/v-pow.html#v-pow
  vdPow(arr_size, vec_base, vec_exp, vec_out);
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}

double my_pow(uint64_t arr_size, double *restrict vec_base,
              double *restrict vec_exp, double *restrict vec_out) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();

  for (uint64_t i = 0; i < arr_size; ++i) {
    vec_out[i] = pow(vec_base[i], vec_exp[i]);
  }
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}

double my_dot_product(double *vec_a, double *vec_b, double *sum,
                      int64_t arr_size) {
  return 0;
}
