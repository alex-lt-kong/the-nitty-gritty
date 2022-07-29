#include <stdio.h>

int main(){

  short a = 32765;
  FILE *fp;
  fp = fopen("integer-overflow.out","w");

  for (int i=0; i<5; i++) {
    a++;
    printf("%d\n", a);
    fprintf(fp,"%d\n",a);
  }

  fclose(fp);
  return 0;
}