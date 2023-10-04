#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int function(int depth) {
  int tmp = rand() % 65536;
  tmp = tmp - 1;
  printf("val: %d; addr: %p; depth: %d\n", tmp, (void*)&tmp, depth);
  tmp = function(++depth) + 1;
  return tmp;
}
int main() {
  srand(time(NULL));
  int res = function(0);
  printf("%d\n", res);
  return 0;
}