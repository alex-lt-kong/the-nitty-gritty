#include <stdio.h>
#include <stdlib.h>

size_t add(size_t a, size_t b) {
    size_t sum = 0;
    sum = a + b;
    return sum;
}

size_t multiply(size_t multiplicand, size_t multiplier) {
    size_t product = 0;
    for (size_t i = 0; i < multiplier; ++i  ) {
        product = add(product, multiplicand);
    }
    return product;
}

int main() {
    size_t a = 12, b = 34;
    size_t product;
    product = multiply(a, b);
    printf("%u * %u = %u\n", a, b, product);
    return 0;
}
