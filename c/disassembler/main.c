#include <stdio.h>
#include <stdlib.h>

#include "sum.c"

int main() {
    int a = 3, b = 12345;
    printf("sum(%d, %d): %d\n", a, b, sum(a, b));
    return 0;
}