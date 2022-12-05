#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(){
  uint8_t* ptr8;
  int32_t* ptr32;
  printf("sizeof(ptr8): %lu, sizeof(ptr32): %lu\n", sizeof(ptr8), sizeof(ptr32));
  return 0;
}