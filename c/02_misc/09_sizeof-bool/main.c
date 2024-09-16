#include <stdbool.h>
#include <stdio.h>

int main() {
  bool b = true;
  printf("b: %d, sizeof(b): %ld\n", b, sizeof(b));
  char c = 'c';
  printf("c: %c, sizeof(c): %ld\n", c, sizeof(c));
  return 0;
}