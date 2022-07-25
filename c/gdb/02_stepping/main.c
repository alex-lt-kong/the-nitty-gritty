#include <stdio.h>
#include <stdlib.h>

int sum_them_all(int* arr, int arr_len) {
    int sum = 0;
    for (int i = 0; i < arr_len; ++i) {
        sum = sum + arr[i];
    }
    return sum;
}

int main() {    
    int arr[] = {3,1,4,1,5,9,2,6};
    printf("sum_them_all(): %d\n", sum_them_all(arr, sizeof(arr)/sizeof(arr[0])));
    return 0;
}