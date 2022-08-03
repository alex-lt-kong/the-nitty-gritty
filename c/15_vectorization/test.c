#include "utils.h"

int main() {
    // Test cases from: https://www.calculator.net/standard-deviation-calculator.html?numberinputs=10%2C+12%2C+23%2C+23%2C+16%2C+23%2C+21%2C+16&ctype=s&x=56&y=16
    double arr0[] = {10, 12, 23, 23, 16, 23, 21, 16};
    printf("result: %lf\nexpect: 5.237229\n", standard_deviation(arr0, sizeof(arr0)/sizeof(arr0[0]), true));
    printf("result: %lf\nexpect: 4.898979\n", standard_deviation(arr0, sizeof(arr0)/sizeof(arr0[0]), false));
    double arr1[] = {101, 12, 223, 23, 12346, 345623, 221, 0, 9527};
    printf("result: %lf\nexpect: 114370.6847\n", standard_deviation(arr1, sizeof(arr1)/sizeof(arr1[0]), true));
    printf("result: %lf\nexpect: 107829.71562\n", standard_deviation(arr1, sizeof(arr1)/sizeof(arr1[0]), false));
    double arr2[] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };
    printf("result: %lf\nexpect: 0\n", standard_deviation(arr2, sizeof(arr2)/sizeof(arr2[0]), true));
    printf("result: %lf\nexpect: 0\n", standard_deviation(arr2, sizeof(arr2)/sizeof(arr2[0]), false));
    double arr3[] = {3.14, 1.414};
    printf("result: %lf\nexpect: 1.22046\n", standard_deviation(arr3, sizeof(arr3)/sizeof(arr3[0]), true));
    printf("result: %lf\nexpect: 0.863  \n", standard_deviation(arr3, sizeof(arr3)/sizeof(arr3[0]), false));
    double arr4[] = {2147483647, 0, -12345, 0.0001,  2.718281828};
    printf("result: %lf\nexpect: 960385262.97615\n", standard_deviation(arr4, sizeof(arr4)/sizeof(arr4[0]), true));
    printf("result: %lf\nexpect: 858994693.04147\n", standard_deviation(arr4, sizeof(arr4)/sizeof(arr4[0]), false));
    double arr5[] = {
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,
        4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811
    };
    printf("result: %lf\nexpect: 70598.361415\n", standard_deviation(arr5, sizeof(arr5)/sizeof(arr5[0]), true));
    printf("result: %lf\nexpect: 69370.47015\n", standard_deviation(arr5, sizeof(arr5)/sizeof(arr5[0]), false));
    return 0;
    
}