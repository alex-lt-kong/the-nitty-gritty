#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void function(int depth) {
  int tmp = rand();
  printf("current depth: %d\n", depth);
  function(++depth);
}

int main() {
  srand(time(NULL));
  function(0);
  return 0;
}

