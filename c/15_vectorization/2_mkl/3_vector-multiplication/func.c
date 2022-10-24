#include </opt/intel/oneapi/mkl/latest/include/mkl.h>

#include "func.h"

uint64_t get_timestamp_in_microsec() {
    struct timeval tv;
    gettimeofday(&tv, NULL); 
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

double mkl_dot_product(double *restrict vec_a, double *restrict vec_b, double *restrict sum, int64_t arr_size)
{
    uint64_t t0, t1;
    t0 = get_timestamp_in_microsec();
    // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-dot.html
    *sum = cblas_ddot(arr_size, vec_a, 1, vec_b, 1);
    t1 = get_timestamp_in_microsec();
    return (t1 - t0) / 1000.0;
}

void* _my_dot_product_job(void* payload) {
    struct JobPayload* pl = (struct JobPayload*)payload;
    for (int i = 0; i < pl->arr_size; ++i) {
        *(pl->sum) += pl->vec_a[i] * pl->vec_b[i];
    }
    return NULL;
}

double my_dot_product(double* vec_a, double* vec_b, double* sum, int64_t arr_size)
{
    uint64_t t0, t1;
    t0 = get_timestamp_in_microsec();
    if (arr_size <= 1 * 1000) {
        for (int i = 0; i < arr_size; ++i) {
            *sum += vec_a[i] * vec_b[i];
        }
    } else {
        struct JobPayload pl0, pl1;
        pthread_t tid0, tid1, tid2;

        pl0.vec_a = vec_a;
        pl0.vec_b = vec_b;
        pl0.sum = calloc(sizeof(double), 1);
        pl0.arr_size = arr_size / 2;

        pl1.vec_a = vec_a + arr_size * sizeof(double) / 2;
        pl1.vec_b = vec_b + arr_size * sizeof(double) / 2;
        pl1.sum = calloc(sizeof(double), 1);
        pl1.arr_size = arr_size - arr_size / 2;
        printf("%lu\n",  pl1.vec_a -  pl0.vec_a);
        pthread_create(&tid0, NULL, _my_dot_product_job, (void*)&pl0);
        pthread_create(&tid1, NULL, _my_dot_product_job, (void*)&pl1); // WRONG!!!
        pthread_join(tid0, NULL);
        pthread_join(tid1, NULL);
        *sum = *pl0.sum + *pl1.sum; 
    }
    t1 = get_timestamp_in_microsec();
    return (t1 - t0) / 1000.0;
}
