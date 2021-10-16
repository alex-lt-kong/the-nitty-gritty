#include <stdio.h>
#include <stdlib.h>

int main() {

  FILE *fp;
  fp = fopen("invert-matrix.in","w");
  int size = 200 * 200;
  int upper = 1024;
  for (int i = 0; i < size; ++i)
  {
    fprintf(fp, "%d\n", rand() % upper);
  }
  fprintf(fp, "\n");
  fclose(fp);
  return 0;
}