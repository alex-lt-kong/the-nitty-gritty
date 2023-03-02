#include <stdio.h>
#include <stdlib.h>

int main() {
    float pi = 3.14;
    float* pi_ptr = &pi; // well-defined
    printf("pi: %f, %f\n", pi, *pi_ptr);
    *pi_ptr = 4.13;
    printf("pi: %f, %f\n", pi, *pi_ptr);
    unsigned int* pi_int = (unsigned int*)&pi; // UB!!!
    printf("pi: %f, %f, %u\n", pi, *pi_ptr, * pi_int);
    return 0;
}