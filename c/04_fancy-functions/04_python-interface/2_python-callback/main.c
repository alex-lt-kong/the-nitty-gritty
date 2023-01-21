#include<stdlib.h>
#include<stdio.h>
#include<stdint.h>
#include<time.h>
#include"func.h"

#define ARR_SIZE 1000000000UL
#define SAMPLE_SIZE 10

int64_t apply(int64_t a) {
    return a + 1;
}

int main() {
    srand(time(NULL));
    uint64_t* arr = malloc(sizeof(uint64_t) * ARR_SIZE);
    uint64_t sample_idxes[SAMPLE_SIZE];
    if (arr == NULL) {
        fprintf(stderr, "malloc() failed\n");
        return EXIT_FAILURE;
    }
    for (uint64_t i = 0; i < ARR_SIZE; ++i) {
        arr[i] = rand();
    }
    for (uint64_t i = 0; i < 10; ++i) {
        sample_idxes[i] = rand() % ARR_SIZE;
    }
    printf("%lu-element array prepared.\n%d samples are:    ",
        ARR_SIZE, SAMPLE_SIZE);
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        printf("%lu, ", arr[sample_idxes[i]]);
    }
    printf("\n");
    struct timespec start, end;
    timespec_get(&start, TIME_UTC);
    manipulate_inplace(arr, ARR_SIZE, &apply);
    timespec_get(&end, TIME_UTC);
    int64_t diff_ns = end.tv_sec * 1000000000 + end.tv_nsec - (
        start.tv_sec * 1000000000 + start.tv_nsec);
    printf("%d samples become: ", SAMPLE_SIZE);
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        printf("%lu, ", arr[sample_idxes[i]]);
    }
    printf("\n");
    free(arr);
    printf("Calling back %luM times takes: %lf sec (%lfK / sec)\n",
        ARR_SIZE / 1000000, diff_ns / 1000.0 / 1000.0 / 1000.0,
        ARR_SIZE / (diff_ns / (1000.0 * 1000.0)));
    return 0;
}
