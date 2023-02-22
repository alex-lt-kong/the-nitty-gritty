#include <stdio.h>

int main(){

  unsigned int a = 4294967250;
  for (int i=0; i<100; ++i) {
    a++;
    printf("%u\n", a);
  }
  return 0;
}