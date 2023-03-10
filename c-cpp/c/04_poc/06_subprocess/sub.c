#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {

    const size_t iter_count = 5;
    fprintf(stderr, "The 1st line from fprintf(stderr, ...)\n");
    for (size_t i = 0; i < iter_count; ++i) {
        printf("[%lu/%lu] This line is from printf()\n", i+1, iter_count);
    }
    fprintf(stderr, "The 2nd line from fprintf(stderr, ...)\n");
    
    if (argc == 2 && strcmp(argv[1], "segfault") == 0)
    {
        int arr[5] = {3,1,4,1,5};
        printf("%d\n", arr[65536]);
    }
    if (argc == 2 && strcmp(argv[1], "flooding") == 0)
    {
        const size_t flooding_iter_count = 65536;
        for (size_t i = 0; i < flooding_iter_count; ++i) {
            printf("A lot of data are being sent to stdout: [%u/%u]\n", i, flooding_iter_count);
            fprintf(stderr, "A lot of data are being sent to stderr: [%u/%u]\n",
                i, flooding_iter_count);
        }
    }

    return 0;
}