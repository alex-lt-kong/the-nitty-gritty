#include <stdio.h>

int main(){

  FILE *fp;
  fp = fopen("int-to-char.out","w");
  for (int i=70; i<80; i++) {
    printf("as int: %d, as char: %c, addr: %p, %ld\n", i, i, &i, &i);
    fprintf(fp, "as int: %d, as char: %c, addr: %p, %ld\n", i, i, &i, &i);
    // addr canNOT use %d, possibly because 64bit machines need more than int
    // to display the address.
  }

  return 0;
}