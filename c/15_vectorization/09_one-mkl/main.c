
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include </opt/intel/oneapi/mkl/2022.1.0/include/mkl.h>

#define SIZE 1500 * 1000 * 1000

/*below is the declaration as given in header file*/
/*float cblas_sasum(const MKL_INT N, const float *X, const MKL_INT incX);*/

uint64_t get_timestamp_in_microsec() {
    struct timeval tv;
    gettimeofday(&tv, NULL); 
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

int main()
{
    float max_float = 1;
    double result0 = 0, result1 = 0;
    uint64_t t0, t1, t2;
    double* arr = malloc(SIZE * sizeof(double));
    printf("Generating %d random double...\n", SIZE);
    for (int j = 0; j < SIZE; ++j) {
        arr[j] = (double)rand()/(double)(RAND_MAX / max_float);
    }
    printf("generated\n");

    t0 = get_timestamp_in_microsec();
    result0 = cblas_dasum(SIZE, arr, 1);
    t1 = get_timestamp_in_microsec();
    printf("result0: %lf from mkl, takes %lu ms\n", result0, (t1 - t0) / 1000);
    t1 = get_timestamp_in_microsec();
    for (int j = 0; j < SIZE; ++j) {
        result1 +=  arr[j];
    }
    t2 = get_timestamp_in_microsec();
    printf("result1: %lf from gcc, takes %lu ms\n", result1, (t2 - t1) / 1000);
    return 0;
}