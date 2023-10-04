#include <stddef.h>
#include <stdint.h>

void manipulate_inplace_int(int* arr, int* y, size_t arr_size) {
    for (int i = 0; i < arr_size; ++i)
        arr[i] = *y + 42;
}

void manipulate_inplace_short(int* arr, int16_t* y, size_t arr_size) {
    for (int i = 0; i < arr_size; ++i)
        arr[i] = *y + 42;
}
