#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

int main() {
    uint32_t* arr;
    clock_t start_time;
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
    fprintf(fp, "Mem Size (Kb),Time (us)\n");
    for (size_t arr_len = 256; arr_len < 1024 * 1024 * 1024; arr_len *= 2) {        
        arr = malloc(sizeof(uint32_t) * arr_len);
        size_t length_mod = arr_len - 1;
        size_t iter = 256;
        size_t step = arr_len / 256;
        start_time = clock();
        for (size_t i = 0; i < arr_len; i += step) {
            arr[(i * 16) & length_mod]++;
        }
        fprintf(fp, "%.01lf,%ld\n", arr_len * 4 / 1024.0, clock() - start_time);
        free(arr);
    }
    fclose(fp);
    return 0;
}