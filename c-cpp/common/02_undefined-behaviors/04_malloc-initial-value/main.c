#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {
  size_t dim[] = {100, 100, 100, 100, 100};
  int* arr;
  for (int i = 0; i < sizeof(dim)/sizeof(dim[0]); ++i) {
    printf("%d-th iter:\n", i);
    arr = malloc(dim[i] * 4);
    for (int j = 0; j < dim[i]; ++j) {
      printf("%d, ", arr[j]);
      ++ arr[j];
    }
    printf("\n");
    free(arr);
  }
  
  return 0;
}