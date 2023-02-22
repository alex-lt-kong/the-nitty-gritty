#include <stdio.h>
#include <stdlib.h>

int divide_by_2(int a) {
    int result = a / 2;
    return result;
}

int divide_by_constant(int a) {
    int result = a / 13;
    return result;
}

int main() {
    int a = 31415;
    printf("divide_by_2(%d): %d\n", a, divide_by_2(a));
    printf("divide_by_constant(%d): %d\n", a, divide_by_constant(a));
    return 0;
}