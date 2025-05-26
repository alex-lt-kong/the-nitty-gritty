#ifndef INC_3_MY_GLOBAL_SCOPED_NEW_AND_DELETE_H
#define INC_3_MY_GLOBAL_SCOPED_NEW_AND_DELETE_H

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <iostream>

static int new_delete_calls_diff = 0;
static int new_delete_array_calls_diff = 0;

void *operator new(std::size_t size) {
    ++new_delete_calls_diff;
    if (void *ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete(void *ptr) noexcept {
    std::free(ptr);
    --new_delete_calls_diff;
}

void *operator new[](std::size_t size) {
    ++new_delete_array_calls_diff;
    if (void *ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}


void operator delete[](void *ptr) noexcept {
    --new_delete_array_calls_diff;
    std::free(ptr);
}

#endif // INC_3_MY_GLOBAL_SCOPED_NEW_AND_DELETE_H
