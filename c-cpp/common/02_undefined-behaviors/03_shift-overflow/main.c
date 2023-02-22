#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {
  uint8_t a = 1;
  for (int i = 0; i < 33; ++i) {
    printf("%u\n", a << i);  
  }
  printf("\n\n=====\n\n");
  a = 1;
  for (int i = 0; i < 35; ++i) {
    printf("Function result: %" PRIu64 "\n", a << i);
  }

  printf("\n\n=====\n\n");
  a = 1;
  for (int i = 0; i < 35; ++i) {
    printf("Function result: %" PRIu64 "\n", (uint64_t)a << i);
  }
  return 0;
}