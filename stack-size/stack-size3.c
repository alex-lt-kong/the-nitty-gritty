#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <time.h>

void function(int depth) {

  int tmp0 = rand();
  int tmp1 = rand();
  long tmp2 = rand();
  printf("current depth: %d\n", depth);
  function(++depth);
}

int main() {

  srand(time(NULL));   // Initialization, should only be called once.
  function(0);
  return 0;
}