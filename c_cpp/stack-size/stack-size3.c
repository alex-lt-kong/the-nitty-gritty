#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int function(int depth) {
  int tmp0 = rand() % 65536;
  int tmp1 = rand() % 65536;
  int tmp2 = rand() % 65536;
  int tmp3 = rand() % 65536;
  int tmp4 = rand() % 65536;
  int tmp5 = rand() % 65536;
  tmp0 = tmp0 - 1;
  tmp1 = tmp1 + 1;
  tmp2 = tmp2 - 2;
  tmp3 = tmp3 + 2;
  tmp4 = tmp4 - 3;
  tmp5 = tmp5 + 3;
  printf("val: %d, %d, %d; addr: %p, %p, %p; depth: %d\n", tmp0, tmp1, tmp2, (void*)&tmp0, (void*)&tmp1, (void*)&tmp2, depth);
  tmp1 = function(++depth);
  return tmp0 - tmp1 + tmp2 - tmp3 + tmp4;
}
int main() {
  srand(time(NULL));
  long res = function(0);
  printf("%d\n", res);
  return 0;
}