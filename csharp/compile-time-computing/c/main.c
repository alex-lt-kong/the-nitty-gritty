#include <stdlib.h>

int Factorial(int n) {
    int result = 1;
    while (n > 1) {
        result *= n--;
    }
    return result;
}

int main() {
    int f4 = Factorial(4);  // f4 == 24
    return 0;
}
