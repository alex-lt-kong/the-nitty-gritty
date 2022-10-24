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
    #if defined( __INTEL_COMPILER)
    #pragma ivdep
    // Pragmas are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.
    // https://stackoverflow.com/questions/5078679/what-is-the-scope-of-a-pragma-directive
    #elif defined(__GNUC__)
    #pragma GCC ivdep
    #endif
    for (int64_t i = 0; i < pl->arr_size; ++i) {
        pl->sum += pl->vec_a[i + pl->offset] * pl->vec_b[i + pl->offset];
    }
    return NULL;
}

double my_dot_product(double* vec_a, double* vec_b, double* sum, int64_t arr_size)
{
    uint64_t t0, t1;
    t0 = get_timestamp_in_microsec();
    *sum = 0;
    if (arr_size < 100 * 1000) {
        struct JobPayload pl0;        
        pl0.vec_a = vec_a;
        pl0.vec_b = vec_b;
        pl0.sum = 0;
        pl0.arr_size = arr_size;
        pl0.offset = 0;
        _my_dot_product_job(&pl0);
        *sum = pl0.sum;
    } else if (arr_size <= 1000 * 1000) {
        struct JobPayload pl0, pl1;
        pthread_t tid0, tid1;

        pl0.vec_a = vec_a;
        pl0.vec_b = vec_b;
        pl0.sum = 0;
        pl0.arr_size = arr_size / 4;
        pl0.offset = 0;
        pthread_create(&tid0, NULL, _my_dot_product_job, (void*)&pl0); // takes < 100 us to pthread_create() a thread

        pl1.vec_a = vec_a;
        pl1.vec_b = vec_b;
        pl1.sum = 0;
        pl1.arr_size = arr_size / 4;
        pl1.offset = arr_size / 4 * 1;
        pthread_create(&tid1, NULL, _my_dot_product_job, (void*)&pl1);        

        pthread_join(tid0, NULL);
        pthread_join(tid1, NULL);
        *sum = pl0.sum + pl1.sum; 
    }else if (arr_size <= 10000 * 1000 * 1000) {
        struct JobPayload pl0, pl1, pl2, pl3;
        pthread_t tid0, tid1, tid2, tid3;

        pl0.vec_a = vec_a;
        pl0.vec_b = vec_b;
        pl0.sum = 0;
        pl0.arr_size = arr_size / 4;
        pl0.offset = 0;
        pthread_create(&tid0, NULL, _my_dot_product_job, (void*)&pl0); // takes < 100 us to pthread_create() a thread

        pl1.vec_a = vec_a;
        pl1.vec_b = vec_b;
        pl1.sum = 0;
        pl1.arr_size = arr_size / 4;
        pl1.offset = arr_size / 4 * 1;
        pthread_create(&tid1, NULL, _my_dot_product_job, (void*)&pl1);

        pl2.vec_a = vec_a;
        pl2.vec_b = vec_b;
        pl2.sum = 0;
        pl2.arr_size = arr_size / 4;
        pl2.offset = arr_size / 4 * 2;
        pthread_create(&tid2, NULL, _my_dot_product_job, (void*)&pl2);

        pl3.vec_a = vec_a;
        pl3.vec_b = vec_b;
        pl3.sum = 0;
        pl3.arr_size = arr_size / 4;
        pl3.offset = arr_size / 4 * 3;
        pthread_create(&tid3, NULL, _my_dot_product_job, (void*)&pl3);
        

        pthread_join(tid0, NULL);
        pthread_join(tid1, NULL);
        pthread_join(tid2, NULL);
        pthread_join(tid3, NULL);
        *sum = pl0.sum + pl1.sum + pl2.sum + pl3.sum; 
    } else {
        struct JobPayload pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7;
        pthread_t tid0, tid1, tid2, tid3, tid4, tid5, tid6, tid7;

        pl0.vec_a = vec_a;
        pl0.vec_b = vec_b;
        pl0.sum = 0;
        pl0.arr_size = arr_size / 8;
        pl0.offset = 0;
        pthread_create(&tid0, NULL, _my_dot_product_job, (void*)&pl0); // takes < 100 us to pthread_create() a thread

        pl1.vec_a = vec_a;
        pl1.vec_b = vec_b;
        pl1.sum = 0;
        pl1.arr_size = arr_size / 8;
        pl1.offset = arr_size / 8 * 1;
        pthread_create(&tid1, NULL, _my_dot_product_job, (void*)&pl1);

        pl2.vec_a = vec_a;
        pl2.vec_b = vec_b;
        pl2.sum = 0;
        pl2.arr_size = arr_size / 8;
        pl2.offset = arr_size / 8 * 2;
        pthread_create(&tid2, NULL, _my_dot_product_job, (void*)&pl2);

        pl3.vec_a = vec_a;
        pl3.vec_b = vec_b;
        pl3.sum = 0;
        pl3.arr_size = arr_size / 8;
        pl3.offset = arr_size / 8 * 3;
        pthread_create(&tid3, NULL, _my_dot_product_job, (void*)&pl3);
        pl0.vec_a = vec_a;
        pl0.vec_b = vec_b;
        pl0.sum = 0;
        pl0.arr_size = arr_size / 8;
        pl0.offset = 0;
        pthread_create(&tid0, NULL, _my_dot_product_job, (void*)&pl0); // takes < 100 us to pthread_create() a thread

        pl1.vec_a = vec_a;
        pl1.vec_b = vec_b;
        pl1.sum = 0;
        pl1.arr_size = arr_size / 8;
        pl1.offset = arr_size / 8 * 1;
        pthread_create(&tid1, NULL, _my_dot_product_job, (void*)&pl1);

        pl2.vec_a = vec_a;
        pl2.vec_b = vec_b;
        pl2.sum = 0;
        pl2.arr_size = arr_size / 8;
        pl2.offset = arr_size / 8 * 2;
        pthread_create(&tid2, NULL, _my_dot_product_job, (void*)&pl2);

        pl3.vec_a = vec_a;
        pl3.vec_b = vec_b;
        pl3.sum = 0;
        pl3.arr_size = arr_size / 8;
        pl3.offset = arr_size / 8 * 3;
        pthread_create(&tid3, NULL, _my_dot_product_job, (void*)&pl3);

        pl4.vec_a = vec_a;
        pl4.vec_b = vec_b;
        pl4.sum = 0;
        pl4.arr_size = arr_size / 8;
        pl4.offset = arr_size / 8 * 4;
        pthread_create(&tid4, NULL, _my_dot_product_job, (void*)&pl4); // takes < 100 us to pthread_create() a thread

        pl5.vec_a = vec_a;
        pl5.vec_b = vec_b;
        pl5.sum = 0;
        pl5.arr_size = arr_size / 8;
        pl5.offset = arr_size / 8 * 5;
        pthread_create(&tid5, NULL, _my_dot_product_job, (void*)&pl5);

        pl6.vec_a = vec_a;
        pl6.vec_b = vec_b;
        pl6.sum = 0;
        pl6.arr_size = arr_size / 8;
        pl6.offset = arr_size / 8 * 6;
        pthread_create(&tid6, NULL, _my_dot_product_job, (void*)&pl6);

        pl7.vec_a = vec_a;
        pl7.vec_b = vec_b;
        pl7.sum = 0;
        pl7.arr_size = arr_size / 8;
        pl7.offset = arr_size / 8 * 7;
        pthread_create(&tid7, NULL, _my_dot_product_job, (void*)&pl7);
        

        pthread_join(tid0, NULL);
        pthread_join(tid1, NULL);
        pthread_join(tid2, NULL);
        pthread_join(tid3, NULL);
        pthread_join(tid4, NULL);
        pthread_join(tid5, NULL);
        pthread_join(tid6, NULL);
        pthread_join(tid7, NULL);
        *sum = pl0.sum + pl1.sum + pl2.sum + pl3.sum + pl4.sum + pl5.sum + pl6.sum + pl7.sum; 
    }
    t1 = get_timestamp_in_microsec();
    return (t1 - t0) / 1000.0;
}
