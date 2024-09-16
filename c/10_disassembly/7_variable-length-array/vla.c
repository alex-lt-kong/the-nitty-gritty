#include <stdio.h>

#include "vla.h"

void print_vla(size_t arr_len) {
    int arr[arr_len];
    arr[0] = 0;
    arr[1] = 1;
    for (size_t i = 2; i < arr_len; ++i) {
        arr[i] = arr[i-1] + arr[i-2];
    }
    for (size_t i = 0; i < arr_len; ++i) {
        printf("%d, ", arr[i]);
    }
    printf("\n");
    return;
}
