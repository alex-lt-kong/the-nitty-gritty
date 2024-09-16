#include <stdio.h>

int main() {

  int a = 2147483600;
  for (int i=0; i<100; ++i) {
    a++;
    printf("%d\n", a);
  }
  return 0;
}