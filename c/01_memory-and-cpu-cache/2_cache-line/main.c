#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define ITER 32

int main() {
    srand(time(NULL));
    const size_t arr_len = 256 * 1024 * 1024;   
    clock_t start_time;
    uint32_t* array;
    FILE *fp;
    #if defined( __INTEL_COMPILER)
    fp = fopen("results-icc.csv", "w");
    #elif defined(__GNUC__)
    fp = fopen("results-gcc.csv", "w");
    #else
    fp = fopen("results-unknown.csv", "w");
    #endif    
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file\n");
        return -1;
    }    

    fprintf(fp, "Step,Time,Sum\n");
    int step = 1;
    while (step <= 4096) {
        array = malloc(arr_len * sizeof(uint32_t));
        for (int i = 0; i < arr_len; ++i) {
            array[i] = rand();
        }
        start_time = clock();
        for (int i = 0; i < ITER; ++i) {
            for (int j = 1; j < arr_len; j += step) {
                array[j] += array[j-1];
            }
        }
        long time_elapsed = clock() - start_time;
        fprintf(fp, "%d,%.02lf,%d\n", step, time_elapsed / ITER / 1000.0, array[rand() % arr_len]);
        free(array);
        if (step < 16) {
            ++step;
        } else {
            step *= 2;
        }
    }
    fclose(fp); //Don't forget to close the file when finished   
    return 0;
}

