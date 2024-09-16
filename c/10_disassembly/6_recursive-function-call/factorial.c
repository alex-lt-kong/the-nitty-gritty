#include "factorial.h"

size_t factorial(size_t a) {
    if (a == 1) {
        return a;
    }
    else {
        return a * factorial(a - 1);
    }
};