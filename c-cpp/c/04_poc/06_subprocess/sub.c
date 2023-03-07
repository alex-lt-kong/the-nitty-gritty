#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    const size_t iter_count = 5;
    
    fprintf(stderr, "The 1st line from fprintf(stderr, ...)\n");
    for (size_t i = 0; i < iter_count; ++i) {
        printf("[%lu/%lu] This line is from printf()\n", i+1, iter_count);
    }
    fprintf(stderr, "The 2nd line from fprintf(stderr, ...)\n");
    return 0;
}