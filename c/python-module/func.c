#include <stdio.h>

int square(int i) {
	return i * i;
}

int sum(int* arr, int arr_len) {
    int sum = 0;
    for (int i = 0; i < arr_len; ++i) {
        sum += arr[i];
    }
    return sum;
}

int bubble_sort_inplace(int *arr, size_t arr_len) {
    for (int i = 0; i < arr_len; ++i) {
        for (int j = i; j < arr_len; ++j) {
            if (arr[i] >= arr[j]) { continue;  }
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }
    return 0;
}