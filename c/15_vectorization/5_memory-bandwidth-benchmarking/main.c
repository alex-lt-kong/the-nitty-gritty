#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../utils.h"

#define SIZE_KB {8, 16, 24, 28, 32, 36, 40, 48, 64, 128, 256, 384, 512, 768, 1024, 1025, 2048, 4096, 8192, 16384, 200000}
#define TESTMEM 10000000000 // Approximate, in bytes
#define BUFFERS 1

double timer(void)
{
    struct timeval ts;
    double ans;

    gettimeofday(&ts, NULL);
    ans = ts.tv_sec + ts.tv_usec*1.0e-6;

    return ans;
}

int main(int argc, char **argv)
{
    double *x[BUFFERS];
    uint64_t t1, t2;
    int kbsizes[] = SIZE_KB;
    double bandwidth[sizeof(kbsizes)/sizeof(int)];
    int iterations[sizeof(kbsizes)/sizeof(int)];
    double *address[sizeof(kbsizes)/sizeof(int)][BUFFERS];
    int i, j, k;

    for (k = 0; k < sizeof(kbsizes)/sizeof(int); k++)
        iterations[k] = TESTMEM/(kbsizes[k]*1024);

    for (k = 0; k < sizeof(kbsizes)/sizeof(int); k++)
    {
        // Allocate
        x[0] = (double *) malloc(kbsizes[k]*1024);
        address[k][0] = x[0];
        memset(x[0], 0, kbsizes[k]*1024);


        // Measure
        t1 = get_timestamp_in_microsec();
        for (i = 0; i < iterations[k]; i++) {
            memset(x[0], 0xff, kbsizes[k]*1024);
        }
        t2 = get_timestamp_in_microsec();
        bandwidth[k] = (BUFFERS*kbsizes[k]*iterations[k])/1024.0/1024.0 *1000 * 1000 / (t2-t1);

        // Free
        free(x[0]);
    }

    printf("TESTMEM = %ld\n", TESTMEM);
    printf("BUFFERS = %d\n", BUFFERS);
    printf("Size (kB)\tBandwidth (GB/s)\tIterations\tAddresses\n");
    for (k = 0; k < sizeof(kbsizes)/sizeof(int); k++)
    {
        printf("%7d\t\t%.2f\t\t\t%d\t\t%x", kbsizes[k], bandwidth[k], iterations[k], address[k][0]);
        for (j = 1; j < BUFFERS; j++)
            printf(", %x", address[k][j]);
        printf("\n");
    }

    return 0;
}