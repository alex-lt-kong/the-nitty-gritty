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
    * In a smart pointer's constructor, memory is allocated; in a smart
    pointer's destructor, memory is deallocated.
    * So when a smart pointer object goes out of scope, following the RAII
    principle, the smart pointer's destructor is called and memory gets
    automatically released.

## `std::unique_ptr<>`

* This is the most basic type of smart pointer. As its name suggests, A
unique_ptr does not share its pointer. It cannot be copied to another
`unique_ptr`, passed by value to a function, or used in any C++ Standard
Library algorithm that requires copies to be made.
    * A `unique_ptr` can be moved though--this means that the ownership of
    the memory resource is transferred to another `unique_ptr` and the
    original `unique_ptr` no longer owns it:

    ```C++
    unique_ptr<int> smart_int_arr2 = std::move(smart_int_arr);
    ```

* As always, the first hellow world program looks sane and simple, but...things
becomes more complicated, when we try to incoporate existing C code into C++.
    * Consider the following example, we want to convert the use of raw pointers
    to the use of smart pointers. We do it by handing over the ownership
    of a raw pointer to a smart pointer and forget the raw pointer:

        ```C++
        int* dynamic_int_arr = (int*)malloc(sizeof(int) * arr_size);
        for (int i = 0; i < arr_size; i++) {
            dynamic_int_arr[i] = i;
        }
        std::unique_ptr<int[]> smart_int_ptr(dynamic_int_arr);
        ```

    * What's wrong? Think about it: for raw pointers, we can either `new` it
    or `malloc()` it with some memory. Following the principle of RAII, an
    object's desctructor will be called when it goes out of scope. When a
    smart pointer's destructor is called, how does it know that it should
    call `delete` or `free()`? No, it doesn't ¯\\\_(ツ)\_\/¯ and it simply
    calls `delete`/`delete[]`.
    * Calling `delete` forcefully against a `malloc()`ed pointer results in
    undefined behaviors.
    * To properly handle this, C++ introduces yet another layer of complexity--
    we need to explicitly let the smart pointer know, by passing it a `deleter`:

    ```C++
        struct FreeDeleter
        {
            void operator()(void *p) const
            {
                std::free(p);
            }
        };

        int* dynamic_int_arr = (int*)malloc(sizeof(int) * arr_size);
        for (int i = 0; i < arr_size; i++) {
            dynamic_int_arr[i] = i;
        }
        std::unique_ptr<int[], FreeDeleter> smart_int_ptr(dynamic_int_arr);
    ```
    * It is also possible to make it a oneliner:

    ```C++
        std::unique_ptr<int[], decltype([](void *p){std::free(p);})>
    ```
    



## References

* [Microsoft - Smart pointers (Modern C++)](https://learn.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=msvc-170)
* [Wikipedia - Smart pointer](https://en.wikipedia.org/wiki/Smart_pointer)
* [CPP Reference - Smart pointers](https://en.cppreference.com/book/intro/smart_pointers)
