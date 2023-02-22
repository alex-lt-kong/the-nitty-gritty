#include <stdio.h>

int add(int a, int b) {
    return (a + b);
}
 
int main() {
    int (*my_add)(int, int);
    my_add = add;
    printf("Address of add(): %p\n", &add);
    printf("Value of my_add:   %p\n", my_add);
    
    int a = 123, b = -1;
    printf("add(): %d, my_add(): %d\n", add(a, b), my_add(a, b));
    return 0;
}