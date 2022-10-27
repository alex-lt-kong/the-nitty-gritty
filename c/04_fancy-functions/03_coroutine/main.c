// C program to demonstrate how coroutines
// can be implemented in C.
#include<stdio.h>
 
/**
 * @brief emulate a range() iterator in C, making it behave like a coroutine
 * 
 * @param a inclusive lower bound
 * @param b exclusive upper bound
 * @return number to be "yield"ed
 */
int range(int a, int b)
{
    if (a <= 0) {
        fprintf(stderr, "This implementation doesn't support any non-positive lower bound...\n");
        return 0;
    }
    static long long int i;
    // the static keyword is the key: it extends the lifetime of the variable.
    // WithOUT static, the lifetime is from the entry into range() to the return from range()
    // With static, the variable's lifetime is the same as the lifetime of the programm.
    // That is, if range() is called again, i's value will be the same as previous call.
    static int state = 0;
    if (state == 0) {
        state = 1;
        i = a;
    }
    while (i < b) {
        printf("\"resumes\" execution@range()\n");
        ++i;
        return (i-1);
    }
    
    state = 0;
    return 0;
}
 
int main() {
    int i;
 
    while (i=range(1, 50)) {
        printf("i is: %d@main()\n", i);
    }
 
    return 0;
}
