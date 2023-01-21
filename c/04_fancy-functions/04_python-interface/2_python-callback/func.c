#include<stdlib.h>
#include<stdint.h>

void manipulate_inplace(uint64_t* arr,
    uint64_t arr_len, int64_t (*func)(int64_t)) {
    for (uint64_t i = 0; i < arr_len; ++i) {
        arr[i] = func(arr[i]);
    }
}
