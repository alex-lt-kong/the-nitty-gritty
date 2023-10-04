#include <stdio.h>
#include <stdlib.h>

int sum(int a, int b) {
    int temp = 0;
    temp = a + b;
    return temp;
}

int main() {
    int a = 3, b = 12345;
    printf("sum(%d, %d): %d\n", a, b, sum(a, b));
    return 0;
}