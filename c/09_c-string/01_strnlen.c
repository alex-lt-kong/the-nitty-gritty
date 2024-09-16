#include <stdio.h>
#include <string.h>

int main() {
  printf("strnlen() of string literal is okay: %ld\n", strnlen("test-string", 32));
  return 0;
}