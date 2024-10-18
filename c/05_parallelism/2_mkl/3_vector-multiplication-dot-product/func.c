#include "func.h"
#include "../../utils.h"

#include <mkl.h>

typedef void (*MathJob)(void*);

// extern uint64_t get_timestamp_in_microsec();

#ifdef _WIN32
__declspec(dllexport)
#endif
double mkl_dot_product(double *restrict vec_a, double *restrict vec_b,
                       double *restrict sum, int64_t arr_size) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();
  // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-dot.html
  *sum = cblas_ddot(arr_size, vec_a, 1, vec_b, 1);
  t1 = get_timestamp_in_microsec();
  return (t1 - t0) / 1000.0;
}

void *_my_dot_product_job(void *payload) {
  struct JobPayload *pl = (struct JobPayload *)payload;
#if defined(__INTEL_COMPILER)
#pragma ivdep
// Pragmas are specific for the compiler and platform in use. So the best bet is
// to look at compiler's documentation.
// https://stackoverflow.com/questions/5078679/what-is-the-scope-of-a-pragma-directive
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
  for (int64_t i = 0; i < pl->arr_size; ++i) {
    pl->sum += pl->vec_a[i + pl->offset] * pl->vec_b[i + pl->offset];
  }
  return NULL;
}

double jobs_dispatcher(double *vec_a, double *vec_b, double *sum, int64_t arr_size,
                  MathJob job_func) {
  uint64_t t0, t1;
  t0 = get_timestamp_in_microsec();
  *sum = 0;
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
    size_t cpu_count = get_cpu_cores();
    if (cpu_count <= 0) {
      cpu_count = 4;
      fprintf(stderr, "get_cpu_cores() failed, default to %lu\n", cpu_count);
	 } 
	thread_count = thread_count > cpu_count ? cpu_count : thread_count;
    struct JobPayload *payloads = malloc(thread_count * sizeof(struct JobPayload));
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
		payloads[i].offset = arr_size / thread_count * i;
	}
	
	for (int i = 0; i < thread_count; ++i) {
	#ifdef _WIN32
          tids[i] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)job_func, (void *)&payloads[i], 0, NULL);
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
	for (int i = 0; i < thread_count; ++i){
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
    double my_dot_product(double *vec_a, double *vec_b, double *sum,
                          int64_t arr_size) {
  return jobs_dispatcher(vec_a, vec_b, sum, arr_size, _my_dot_product_job);
}