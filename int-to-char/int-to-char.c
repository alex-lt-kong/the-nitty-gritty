#include <stdio.h>

int main(){

  for (int i=70; i<80; i++) {
    printf("as int: %d, as char: %c, addr: %ld\n", i, i, &i);
    // addr canNOT use %d, possibly because 64bit machines need more than int
    // to display the address.
  }

  return 0;
}