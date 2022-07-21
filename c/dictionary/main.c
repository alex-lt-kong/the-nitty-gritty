#include <stdio.h>

#include "dict.c"

int main() {
    install("hello", "world!");
    install("foo", "bar");
    install("pi", "3.1415926535");
    printf("%s\n", lookup("hello")->defn);
    printf("%s\n", lookup("foo")->defn);
    printf("%s\n", lookup("foo")->defn);    
    printf("%s\n", lookup("pi")->defn);    
    printf("%s\n", lookup("foo")->defn);
    return 0;
}