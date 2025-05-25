#ifndef INC_2_MY_NEW_AND_DELETE_H
#define INC_2_MY_NEW_AND_DELETE_H

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// Roughly C++20 standard specification for new/delete operators:
// https://timsong-cpp.github.io/cppwp/n4868/basic.stc
// 6.7.5.5.2 Allocation functions
// An allocation function shall be a class member function or a global function;
// a program is ill-formed if an allocation function is declared in a namespace
// scope other than global scope or declared static in global scope


// new/delete are just two operators that we can overload, similar to how we can
// overload operator+ or operator[].
class MyClass {

public:
    static int m_new_delete_calls_diff;
    static int m_new_delete_array_calls_diff;
    int m_data;
    // Funny to note that in new and delete there is no reference to
    // constructor/destructor--they are not called in new/delete.
    // Also, new() takes a size argument, compiler decides what it should be
    void *operator new(const std::size_t size) {
        ++m_new_delete_calls_diff;
        if (sizeof(MyClass) != size)
            throw std::logic_error("sizeof(MyClass) != size");
        return std::malloc(size);
    }

    void operator delete(void *ptr) noexcept {
        std::free(ptr);
        --m_new_delete_calls_diff;
    }

    void *operator new[](const std::size_t size) {
        ++m_new_delete_array_calls_diff;
        return std::malloc(size);
    }

    void operator delete[](void *ptr) noexcept {
        std::free(ptr);
        --m_new_delete_array_calls_diff;
    }

    MyClass() : m_data(0) {}
    MyClass(const int data) : m_data(data) {}
};
int MyClass::m_new_delete_calls_diff = 0;
int MyClass::m_new_delete_array_calls_diff = 0;

#endif // INC_2_MY_NEW_AND_DELETE_H
