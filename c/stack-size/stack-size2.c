#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int function(int depth) {
  int tmp0 = rand() % 65536;
  int tmp1 = rand() % 65536;
  tmp0 = tmp0 - 1;
  printf("val: %d, %d; addr: %p, %p; depth: %d\n", tmp0, tmp1, (void*)&tmp0, (void*)&tmp1, depth);
  tmp1 = function(++depth);
  return tmp1 - tmp0;
}
int main() {
  srand(time(NULL));
  int res = function(0);
  printf("%d\n", res);
  return 0;
}