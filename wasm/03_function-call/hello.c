#include <stdio.h>
#include <emscripten/emscripten.h>

EMSCRIPTEN_KEEPALIVE int sum(int a, int b) {
    return a+b;
}

EMSCRIPTEN_KEEPALIVE void myFunction(int argc, char ** argv) {
    printf("MyFunction Called\n");
}

int main() {
    int a = 3;
    int b = 23;
    printf("Hello World\n");
    printf("The sum of %d and %d is %d\n", a, b, sum(a, b));
    return 0;
}
