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


int partition(int *arr, int lo, int hi) {
    int pos = lo - 1;
    int pivot = arr[hi];
    for (int i = lo; i <= hi; ++i) {
        if (arr[i] < pivot) {
            ++pos;
            int t = arr[i];
            arr[i] = arr[pos];
            arr[pos] = t;
        }
    }
    ++pos;
    int t = arr[hi];
    arr[hi] = arr[pos];
    arr[pos] = t;
    return pos;
}

int quick_sort_inplace(int* arr, int lo, int hi) {
    if (lo < hi) {
        int pos = partition(arr, lo, hi);
        quick_sort_inplace(arr, lo, pos - 1);
        quick_sort_inplace(arr, pos + 1, hi);
    }
}