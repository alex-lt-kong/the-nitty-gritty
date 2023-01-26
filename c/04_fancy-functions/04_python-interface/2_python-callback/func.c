#include<stdlib.h>
#include<stdint.h>
#include<stdio.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
__declspec(dllexport)
#endif
void manipulate_inplace(uint64_t* arr,
    uint64_t arr_len, uint64_t(*func)(uint64_t)) {
    for (uint64_t i = 0; i < arr_len; ++i) {
        arr[i] = func(arr[i]);
    }
}
