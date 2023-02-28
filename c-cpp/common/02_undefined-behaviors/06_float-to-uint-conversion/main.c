#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {

    float    neg_pi = -3.14;
    uint32_t b = (uint32_t)neg_pi;
    printf("Converting %f to uint32_t gives %u\n", neg_pi, b);
    float big_num = 2147483646.0;
    uint8_t  c = (uint8_t )big_num;
    printf("Converting %f to uint8_t gives %u\n", big_num, c);
    return 0;
}