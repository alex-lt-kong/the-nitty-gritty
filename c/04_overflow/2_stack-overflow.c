#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>


void function(int depth) {

  int tmp = 12345;
  printf("current depth: %d\n", depth);
  if (depth > 2147483648)
    return;
  else
    function(++depth);
}

int main() {

  function(0);
  return 0;
}

