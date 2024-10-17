#include "../../utils.h"

#include <mkl.h>

#include <math.h>
#include <stdint.h>

// Somehow this is needed on Linux...
extern inline uint64_t get_timestamp_in_microsec();

#ifdef _WIN32
__declspec(dllexport)
#endif
    double mkl_ln(uint64_t arr_size, double *restrict vec_in,
                  double *restrict vec_out) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();
  // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/vector-mathematical-functions/vm-mathematical-functions/exponential-and-logarithmic-functions/v-ln.html#v-ln
  vdLn(arr_size, vec_in, vec_out);
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    double my_ln(uint64_t arr_size, double *restrict vec_in,
                 double *restrict vec_out) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();

  for (uint64_t i = 0; i < arr_size; ++i) {
    vec_out[i] = log(vec_in[i]);
  }
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}
