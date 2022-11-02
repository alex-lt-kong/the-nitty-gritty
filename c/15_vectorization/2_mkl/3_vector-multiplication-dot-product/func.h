#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct JobPayload {
    double* vec_a;
    double* vec_b;
    double sum;
    int64_t arr_size;
    int64_t offset;
};