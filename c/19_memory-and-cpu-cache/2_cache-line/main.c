#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITER 32

int main() {
    srand(time(NULL));
    const size_t arr_len = 256 * 1024 * 1024;   
    clock_t start_time;
    int* array;
    FILE *fp;
    fp = fopen("results.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file\n");
    }    

    fprintf(fp, "Step,Time,Sum\n");
    int step = 1;
    unsigned int sum;
    while (step <= 65536) {
        array = malloc(arr_len * sizeof(int));
	sum = 0;
        for (int i = 0; i < arr_len; ++i) {
            array[i] = rand() % (RAND_MAX / 2 - 1);
        }
        start_time = clock();
        for (int i = 0; i < ITER; ++i) {
            for (int j = 0; j < arr_len; j += step) {
                sum += array[j];
            }
        }
        long time_elapsed = clock() - start_time;
        fprintf(fp, "%d,%.02lf,%d\n", step, time_elapsed / ITER / 1000.0, sum);
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
