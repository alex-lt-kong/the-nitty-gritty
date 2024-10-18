#include "../../utils.h"
#include "func.h"

#include <mkl.h>

#include <math.h>
#include <stdint.h>

typedef void (*MathJob)(void *);

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

double jobs_dispatcher(double *vec_a, double *vec_b, double *sum,
                       int64_t arr_size, MathJob job_func) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();
  *sum = 0;
  DWORD cpu_count = get_cpu_cores();
  printf("cpu_count: %ul\n", cpu_count);
  if (arr_size < 1000 * 1000) {
    struct JobPayload pl0;
    pl0.vec_a = vec_a;
    pl0.vec_b = vec_b;
    pl0.sum = 0;
    pl0.arr_size = arr_size;
    pl0.offset = 0;
    job_func(&pl0);
    *sum = pl0.sum;
  } else {
    if (arr_size % (1000 * 1000) != 0) {
      fprintf(stderr, "this naive benchmark only accepts arr_size as a "
                      "multiple of 1,000,000...\n");
      abort();
    }
    size_t thread_count = arr_size / (1000 * 1000);

    thread_count = thread_count > 64 ? 64 : thread_count;
    struct JobPayload *payloads =
        malloc(thread_count * sizeof(struct JobPayload));
    if (payloads == NULL) {
      fprintf(stderr, "malloc() failed\n");
      return -1;
    }
#ifdef _WIN32
    HANDLE *tids = malloc(thread_count * sizeof(HANDLE));
#else
    pthread_t *tids = malloc(thread_count * sizeof(pthread_t));
#endif
    if (tids == NULL) {
      fprintf(stderr, "malloc() failed\n");
      free(payloads);
      return -1;
    }

    for (int i = 0; i < thread_count; ++i) {
      payloads[i].vec_a = vec_a;
      payloads[i].vec_b = vec_b;
      payloads[i].sum = 0;
      payloads[i].arr_size = arr_size / thread_count;
      payloads[i].offset = i == arr_size / thread_count * i;
    }

    for (int i = 0; i < thread_count; ++i) {
#ifdef _WIN32
      tids[i] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)job_func,
                             (void *)&payloads[i], 0, NULL);
#else
      // takes < 100 us to pthread_create() a thread
      pthread_create(&tids[i], NULL, job_func, (void *)&payloads[i]);
#endif
    }

    for (int i = 0; i < thread_count; ++i) {
#ifdef _WIN32
      WaitForSingleObject(tids[i], INFINITE);
#else
      pthread_join(tids[i], NULL);
#endif
    }
    *sum = 0;
    for (int i = 0; i < thread_count; ++i) {
      *sum += payloads[i].sum;
    }
    free(payloads);
    free(tids);
  }
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    double my_pow(double *vec_a, double *vec_b, double *sum,
                          int64_t arr_size) {
  return jobs_dispatcher(vec_a, vec_b, sum, arr_size, _my_dot_product_job);
}
