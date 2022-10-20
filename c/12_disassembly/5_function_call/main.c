#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    int temp = 0;
    temp = a + b;
    return temp;
}

int main() {
    int a = 3, b = 12345;
    int sum;
    sum = add(a, b);
    return sum;
}