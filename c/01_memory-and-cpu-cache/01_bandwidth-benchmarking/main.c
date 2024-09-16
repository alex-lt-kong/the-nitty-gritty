#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../utils.h"

#define SIZE_KB {8, 16, 24, 28, 32, 36, 40, 48, 64, 128, 256, 384, 512, 768, 1024, 1025, 2048, 4096, 8192, 16384, 200000}
#define TEST_MEM_IN_BYTES 107374182400 // 100 * 1024 * 1024 * 1024 = 100GB

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
    uint64_t t1, t2;
    int sizes_in_kb[] = SIZE_KB;
    size_t sizes_in_bytes[] = SIZE_KB;
    double bandwidth[sizeof(sizes_in_kb)/sizeof(int)];
    int iterations[sizeof(sizes_in_kb)/sizeof(int)];
    uint8_t *addr[sizeof(sizes_in_kb)/sizeof(int)];



    for (int i = 0; i < sizeof(sizes_in_kb)/sizeof(int); i++) {
        sizes_in_bytes[i] = sizes_in_kb[i] * 1024;
        iterations[i] = TEST_MEM_IN_BYTES / (sizes_in_bytes[i]);
    }

    for (int i = 0; i < sizeof(sizes_in_kb)/sizeof(int); i++)
    {
        // Allocate
        addr[i] = malloc(sizes_in_bytes[i]);        
        memset(addr[i], 0, sizes_in_bytes[i]);
        

        // Measure
        t1 = get_timestamp_in_microsec();
        for (int j = 0; j < iterations[i]; ++j) {
            memset(addr[i], 0xff, sizes_in_bytes[i]);
        }
        t2 = get_timestamp_in_microsec();
        bandwidth[i] = (sizes_in_kb[i]*iterations[i]) / 1024.0 / 1024.0 * 1000 * 1000 / (t2-t1);

        free(addr[i]);
    }

    printf("TESTMEM = %ld MB\n", TEST_MEM_IN_BYTES / 1024 / 1024);
    printf("Size (kB)\tBandwidth (GB/s)\tIterations\tAddresses\n");
    for (int i = 0; i < sizeof(sizes_in_kb)/sizeof(int); i++)
    {
        printf("%7d\t\t%.2f\t\t\t%d\t\t0x%x", sizes_in_kb[i], bandwidth[i], iterations[i], addr[i]);
        printf("\n");
    }

    return 0;
}