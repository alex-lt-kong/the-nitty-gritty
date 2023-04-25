#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ARRAY_SIZE 8

int* ptr;
int* ptr_orig;

int* arr_plus_n(char* caller_name, int n) {
    for (int i = 0; i < n; ++i) {
        ++ptr;
    }
    printf("arr_plus_n() called by [%s], n: %d, ptr: %ls, *ptr: %d\n",
        caller_name, n, ptr, *ptr);
    return ptr;
}

bool init_ptr() {
    int* ptr_orig = (int*)malloc(ARRAY_SIZE * sizeof(int));
    if (ptr_orig == NULL) {
        perror("malloc()");
        return false;
    }
    ptr = ptr_orig;
    printf("Array: ");
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        ptr[i] = ARRAY_SIZE-i;
        printf("%d,", ptr[i]);
    }
    printf("\n");
    return true;
}

void test_one() {
    if (init_ptr()) {
        printf("%d\n\n", (arr_plus_n("pointer", 1))[*arr_plus_n("index", 2)]);
        free(ptr_orig);
    }
}

void test_two() {
    if (init_ptr()) {
        printf("%d\n\n", *(arr_plus_n("pointer", 1) + *arr_plus_n("index", 2)));
        free(ptr_orig);
    }
}


void test_three() {
    if (init_ptr()) {
        printf("%d\n\n", (arr_plus_n("pointer", 2))[*arr_plus_n("index", 1)]);
        free(ptr_orig);
    }
}

void test_four() {
    if (init_ptr()) {
        printf("%d\n\n", *(arr_plus_n("pointer", 2) + *arr_plus_n("index", 1)));
        free(ptr_orig);
    }
}

int main(void) {
    printf("=== Test One ===\n");
    test_one();
    printf("=== Test Two ===\n");
    test_two();
    printf("=== Test Three ===\n");
    test_three();
    printf("=== Test Four ===\n");
    test_four();
    return 0;
}