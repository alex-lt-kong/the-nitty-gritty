#include <stdio.h>
#include <stdlib.h>

int main() {
   // goto output;
    int a;
    a = 10;
    int arr[a];
output:
    printf("%d (%p)\n", a, &a);
}