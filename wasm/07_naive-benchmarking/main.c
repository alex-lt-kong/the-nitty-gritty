
#include<stdio.h>

int partition(int* arr, int lo, int hi) {
  int pivot = arr[hi];
  int pos = lo - 1;
  for (int i = lo; i <= hi; ++i) {
    if (arr[i] < pivot) {
      ++pos;
      int t = arr[pos];
      arr[pos] = arr[i];
      arr[i] = t;
    }
  }
  int t = arr[pos + 1];
  arr[pos + 1] = arr[hi];
  arr[hi] = t;
  return pos + 1;
}

int* quick_sort(int* arr, int lo, int hi) {
  if (lo < hi) { 
    int pi = partition(arr, lo, hi); 

    // Separately sort elements before 
    // partition and after partition 
    quick_sort(arr, lo, pi - 1); 
    quick_sort(arr, pi + 1, hi); 
  }
  return arr;
}

int main(){
  printf("Welcome to the Amazing world of WASM!\n");
  int arr[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 9};
  int arr_len = sizeof(arr) / sizeof(arr[0]);
  for (int i = 0; i < arr_len; ++i) {
    printf("%d, ", arr[i]);
  }
  printf("\n");
  quick_sort(arr, 0, arr_len-1);
  for (int i = 0; i < arr_len; ++i) {
    printf("%d, ", arr[i]);
  }
  printf("\n");
}
