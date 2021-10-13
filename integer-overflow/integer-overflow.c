#include <stdio.h>

int main(){

  short a = 32765;

  for (int i=0; i<5; i++) {
    a++;
    printf("%d\n", a);
  }

  return 0;
}