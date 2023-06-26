#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define ARR_SIZE 4096

unsigned int my_func(uint8_t num) {
    int* arr_ptr = malloc(sizeof(int) * ARR_SIZE);
    memset(arr_ptr, num, ARR_SIZE);
    unsigned int sum = 0;
    for (int i = 0; i < ARR_SIZE; ++i) {
        sum += arr_ptr[i];
    }
    return sum;
}

int main() {
    unsigned int sum = 0;
    for (uint8_t i = 0; i < 64; ++i) {
        sum += my_func(i);
    }
    printf("%u\n", sum);
    return 0;
}