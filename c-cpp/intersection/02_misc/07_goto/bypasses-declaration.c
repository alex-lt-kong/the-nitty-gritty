#include <stdio.h>
#include <stdlib.h>

int main() {
    goto output;
    int a;
    a = 10;
output:
    printf("%d (%p)\n", a, &a);
}