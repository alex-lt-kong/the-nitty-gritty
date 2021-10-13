#include <stdio.h>
#include <stdlib.h>

int main(){

  int* p;
  FILE *fp;
  fp = fopen("dynamic-array.out","w");
  p = calloc(10, sizeof(int));

  for (int i=0; i<10; i++)
    *(p + i) = i;
  for (int i=0; i<10; i++) {
    printf("*(p + %d) = %d\n", i, *(p+i));
    fprintf(fp, "*(p + %d) = %d\n", i, *(p+i));
  }
  free(p);

  printf("\n");
  fprintf(fp, "\n");

  p = calloc(4, sizeof(int) );

  for (int i = 0; i < 4; i++ )
    p[i] = i*i;

  for (int i = 0; i < 4; i++ ) {
    printf("p[%d] = %d\n", i, p[i]);
    fprintf(fp, "p[%d] = %d\n", i, p[i]);
  }
  free(p);

  return 0;
}