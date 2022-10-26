#include <stdio.h>

#include "dict.c"

int main() {
    install("hello", "world!");
    install("foo", "bar");    
    install("pi", "3.1415926535");

    printf("%s\n", lookup("hello")->value);
    printf("%s\n", lookup("foo")->value);
    printf("%s\n", lookup("foo")->value);    
    printf("%s\n", lookup("pi")->value);
    install("foo", "new bar");
    printf("%s\n", lookup("foo")->value);
    printf("%s\n", lookup("foo")->value);
    return 0;
}