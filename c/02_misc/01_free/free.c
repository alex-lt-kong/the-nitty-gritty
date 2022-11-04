#include <stdio.h>
#include <stdlib.h>

int main(){

  printf("Hello world!\n");
  int size = 10;
  int arr[size];
  for (int i = 0; i < size; i++) {
    arr[i] = i * 3;
    printf("%d,", arr[i]);
  }
  printf("\n");
  //free(arr); canNOT free arr; it causes a segmentfault.

  int* p = calloc(size, sizeof(int));
  for (int i = 0; i < size; i++)
    p[i] = i * 3 + 1;
  for (int i=0; i<10; i++) {
    printf("p[%d]=%d\n", i, p[i]);
  }
  free(p);
  return 0;
}