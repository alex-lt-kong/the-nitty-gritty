#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITER 32

int main() {
    srand(time(NULL));
    const size_t arr_len = 2 << 26; // 512MB;    
    clock_t start_time;
    int* array;
    FILE *fp;
    fp = fopen("results.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file\n");
    }    

    fprintf(fp, "Step,Time,Sample Value\n");
    int step = 1;

    while (step <= arr_len) {
        array = malloc(arr_len * sizeof(int));
        for (int i = 0; i < arr_len; ++i) {
            array[i] = rand() % (RAND_MAX / 2 - 1);
        }
        start_time = clock();
        for (int i = 0; i < ITER; ++i) {
            for (int j = 0; j < arr_len; j += step) {
                array[j] += array[i];
            }
        }
        long time_elapsed = clock() - start_time;
        fprintf(fp, "%d,%.02lf,%d\n", step, time_elapsed / ITER / 1000.0, array[rand() % arr_len]);
        printf("%d/%d\n", step, arr_len);
        free(array);
        if (step < 32) {
            step += 1;
        } else if (step < 128) {
            step += 4;
        } else {
            step *= 2;
        }

    }
    fclose(fp); //Don't forget to close the file when finished   
    return 0;
}