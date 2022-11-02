#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define CACHE_SIZE       (32*1024*1024UL) // MUST be a power of 2
#define CACHE_LINE_SIZE  64              // MUST be a power of 2
#define MAX_STEP         8192            // MUST be a power of 2
#define ACCESSES         123435

int main() {
    const uint64_t arr_len = (CACHE_SIZE / CACHE_LINE_SIZE) * MAX_STEP;
    uint8_t* arr;
    struct timespec ts;
    uint8_t sum = 0;
    double delta, t0;
    uint64_t idx;
    uint64_t test_size;
    uint64_t test_size_mask;

    arr = malloc(sizeof(uint8_t) * arr_len);   // Consider something like "mmap()" on Linux or "VirtualAlloc()" on Windows here
    if(arr == NULL) {
        printf("Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    for (uint64_t step = CACHE_LINE_SIZE; step <= MAX_STEP; step += 2) {

        test_size = (CACHE_SIZE / CACHE_LINE_SIZE) * step;
        test_size_mask = test_size - 1;

        timespec_get(&ts, TIME_UTC);
        t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
        idx = 0;
        for(int accesses = 0; accesses < ACCESSES; accesses++) { 
            arr[idx] += step;
            sum += arr[idx];
            //idx = (idx + step) & test_size_mask;
            idx = (idx + step) % test_size;
           // if (((idx + step) & test_size_mask) != ((idx + step) % test_size)) {
           //     printf(
             //       "%lu vs %lu, test_size == %lu\n",
                 //   ((idx + step) & test_size_mask), ((idx + step) % test_size), test_size
               // );
            //}
        }

        timespec_get(&ts, TIME_UTC);
        delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
        printf("%lu, %lf\n", step, delta);
    }   
    free(arr);
    return 0;
}