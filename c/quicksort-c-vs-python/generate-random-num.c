#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000000

int main() {

  FILE *fp;
  fp = fopen("quicksort.in","w");
  int upper = 2147483647;
  for (int i = 0; i < SIZE; ++i) {
    fprintf(fp, "%d\n", rand() % upper);
  }
  fclose(fp);
  return 0;
}