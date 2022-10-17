#include <stdio.h>
#include <stdlib.h>

#include "func.c"

int main() {    
    int arr[] = {3,1,4,1,5,9,2,6};
    int result = sum_them_all(arr, sizeof(arr)/sizeof(arr[0]));
    printf("sum_them_all(): %d\n", result);
    return 0;
}
