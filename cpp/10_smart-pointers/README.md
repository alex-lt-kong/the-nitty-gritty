# Smart pointer

* Smart pointers is a much-hyped function in "modern" C++. They are included
in the Standard Library and they are used to help ensure that programs are
free of memory and resource leaks and are exception-safe.

* In C++, a smart pointer is implemented as a template class that mimics, by
means of operator overloading, the behaviors of a traditional (raw) pointer,
(e.g. dereferencing, assignment) while providing additional memory management
features. 

* Practically, the implementation of smart pointers is based on the idea of
[RAII](../01_raii/):
    * In smart pointers' constructor, memory is allocated; in smart pointers'
    destructor, memory is deallocated.
    * So when a smart pointer object goes out of scope, following the RAII
    principle, the smart pointer's destructor is called and memory gets
    automatically released.

## References

* [Microsoft - Smart pointers (Modern C++)](https://learn.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=msvc-170)
* [Wikipedia - Smart pointer](https://en.wikipedia.org/wiki/Smart_pointer)
* [CPP Reference - Smart pointers](https://en.cppreference.com/book/intro/smart_pointers)