#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../utils.h"

#define SIZE_KB {8, 16, 24, 28, 32, 36, 40, 48, 64, 128, 256, 384, 512, 768, 1024, 1025, 2048, 4096, 8192, 16384, 200000}
#define TEST_MEM_IN_BYTES 10737418240 // 10 * 1024 * 1024 * 1024 = 10GB

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
    double *x;
    uint64_t t1, t2;
    int kbsizes[] = SIZE_KB;
    size_t bsizes[] = SIZE_KB;
    double bandwidth[sizeof(kbsizes)/sizeof(int)];
    int iterations[sizeof(kbsizes)/sizeof(int)];
    double *address[sizeof(kbsizes)/sizeof(int)];



    for (int i = 0; i < sizeof(kbsizes)/sizeof(int); i++) {
        bsizes[i] = kbsizes[i] * 1024;
        iterations[i] = TEST_MEM_IN_BYTES / (bsizes[i]);
    }

    for (int i = 0; i < sizeof(kbsizes)/sizeof(int); i++)
    {
        // Allocate
        x = (double *) malloc(bsizes[i]);
        address[i] = x;
        memset(x, 0, bsizes[i]);
        

        // Measure
        t1 = get_timestamp_in_microsec();
        for (int j = 0; j < iterations[i]; ++j) {
            memset(x, 0xff, bsizes[i]);
        }
        t2 = get_timestamp_in_microsec();
        bandwidth[i] = (kbsizes[i]*iterations[i]) / 1024.0 / 1024.0 * 1000 * 1000 / (t2-t1);

        free(x);
    }

    printf("TESTMEM = %ld MB\n", TEST_MEM_IN_BYTES / 1024 / 1024);
    printf("Size (kB)\tBandwidth (GB/s)\tIterations\tAddresses\n");
    for (int i = 0; i < sizeof(kbsizes)/sizeof(int); i++)
    {
        printf("%7d\t\t%.2f\t\t\t%d\t\t%x", kbsizes[i], bandwidth[i], iterations[i], address[i]);
        printf("\n");
    }

    return 0;
}