
#include <stdio.h>
#ifdef EMSCRIPTEN
#include "emscripten.h"
#endif

#ifdef EMSCRIPTEN
EMSCRIPTEN_KEEPALIVE
#endif
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

#ifdef EMSCRIPTEN
EMSCRIPTEN_KEEPALIVE
#endif
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
