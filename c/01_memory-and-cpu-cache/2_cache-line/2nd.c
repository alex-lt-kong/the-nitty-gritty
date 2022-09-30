#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define MAX_STEP 4096

int main() {
    srand(time(NULL));
    const size_t arr_len = 16 * 1024 * 1024;   
    clock_t start_time;
    //uint32_t* array;
    FILE *fp;
    #if defined( __INTEL_COMPILER)
    fp = fopen("2nd-icc.csv", "w");
    #elif defined(__GNUC__)
    fp = fopen("2nd-gcc.csv", "w");
    #else
    fp = fopen("2nd-unknown.csv", "w");
    #endif    
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file\n");
        return -1;
    }    

    fprintf(fp, "Step,Time,Sum\n");

    size_t steps[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20,
        32, 64, 128, 256, 512, 1024, 2048, 4096
    };
    size_t steps_len = sizeof(steps) / sizeof(steps[0]);
    uint32_t* arr_ptrs[steps_len];
    
    for (int i = 0; i < steps_len; ++i) {
        arr_ptrs[i] = malloc(arr_len * sizeof(uint32_t));
        for (int j = 0; j < arr_len; ++j) {
            arr_ptrs[i][j] = rand();
        }
    }
    for (int i = 0; i < steps_len; ++i) {
        
        start_time = clock();
        for (int j = 0; j < arr_len; j += steps[i]) {
            arr_ptrs[i][j] = start_time + j;
        }
        long time_elapsed = clock() - start_time;
        fprintf(fp, "%lu,%.02lf,%u\n", steps[i], time_elapsed / 1000.0, arr_ptrs[i][rand() % arr_len]);
    }
    for (int i = 0; i < steps_len; ++i) {
        free(arr_ptrs[i]);
    }
    fclose(fp); //Don't forget to close the file when finished   
    return 0;
}
