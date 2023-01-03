#include <stdio.h>
#include <stdlib.h>

int main() {
    int a;
    goto output;
    a = 10;
output:
    printf("%d (%p)\n", a, &a);
}