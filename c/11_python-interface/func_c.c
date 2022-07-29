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

int insertion_sort_inplace(int* arr, size_t arr_len) {
  for (int i = 1; i < arr_len; ++i) {
    int key = arr[i];
    int j = i - 1;

    // Compare key with each element on the left of it until an element smaller than
    // it is found.
    // For descending order, change key<array[j] to key>array[j].
    while (key < arr[j] && j >= 0) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
  return 0;
}


int pick_pivot_index(int *arr, int lo, int hi) {
    if (hi - lo <= 1) {
        return lo;
    }
    int mid = (lo + hi) / 2;
    if (arr[lo] <= arr[mid] && arr[mid] <= arr[hi] || arr[lo] >= arr[mid] && arr[mid] >= arr[hi]) {
        return mid;
    }
    if (arr[mid] <= arr[lo] && arr[lo] <= arr[hi] || arr[mid] >= arr[lo] && arr[lo] >= arr[hi]) {
        return lo;
    }
    return hi;
}


int partition(int *arr, int lo, int hi) {
    int pos = lo - 1;
    int pivot_idx = pick_pivot_index(arr, lo, hi);
    int t = arr[pivot_idx];
    arr[pivot_idx] = arr[hi];
    arr[hi] = t;
    int pivot = arr[hi];
    for (int i = lo; i <= hi; ++i) {
        if (arr[i] < pivot) {
            ++pos;
            t = arr[i];
            arr[i] = arr[pos];
            arr[pos] = t;
        }
    }
    ++pos;
    t = arr[hi];
    arr[hi] = arr[pos];
    arr[pos] = t;
    return pos;
}

int quick_sort_inplace(int* arr, int lo, int hi) {
    const int thres = 7;
    if (hi - lo > thres) {
        int pos = partition(arr, lo, hi);
        quick_sort_inplace(arr, lo, pos - 1);
        quick_sort_inplace(arr, pos + 1, hi);
    } else {
        // The idea is that, if the array is too small, we fallback to insertion sort to boost performance.
        insertion_sort_inplace(arr + lo, hi - lo + 1);
    }
}