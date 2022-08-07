#include <stdio.h>
#include <stdlib.h>

#include "func.c"

int main() {
    int a = 3, b = 12345;
    printf("sum(%d, %d): %d\n", a, b, sum(a, b));
    int arr[] = {3,1,4,1,5,9,2,6};
    printf("sum_them_all(): %d\n", sum_them_all(arr, sizeof(arr)/sizeof(arr[0])));
    return 0;
}