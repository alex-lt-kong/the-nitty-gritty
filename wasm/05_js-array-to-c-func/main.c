
#include<stdio.h>

int main(){
  printf("Welcome to the Amazing world of WASM!\n");
}

int* bubble_sort(int* arr, int arr_len) {
  // Even if we sort the array in-place, we still need to return it given how the function is called from
  // the JavaScript side.
  for (int i = 0; i < arr_len; ++i) {
    for (int j = i; j < arr_len; ++j) {
      if (arr[i] < arr[j]) {
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
    }
  }
  return arr;
}