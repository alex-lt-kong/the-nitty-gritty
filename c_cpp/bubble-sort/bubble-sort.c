#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main() {

  int size = 200 * 200, upper = 65536;
  int arr[size];
  int t;

  for (int i = 0; i < size; i++) {
    arr[i] = rand() % upper;
  }

  struct timeval stop, start;
  gettimeofday(&start, NULL);
  for (int i = 0; i < size; i++) {
    for (int j = i; j < size; j++) {
      if (arr[i] < arr[j]) {
        t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
    }
  }
  gettimeofday(&stop, NULL);  
  for (int i = 0; i < size; i++) {
    printf("%d, ", arr[i]);
  }

  printf("%d ms\n", ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000);
  return 0;
}