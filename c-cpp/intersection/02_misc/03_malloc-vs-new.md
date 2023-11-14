# malloc() vs new

1. `new` calls constructors, while `malloc()` does not.
1. `new` type-safe (i.e., it returns exact data type), while `malloc()`
   returns `void *`.
1. `new` throws `std::bad_alloc` on allocation failure while `malloc()`
   returns `NULL`.
1. Reallocation of memory not handled by `new` while `malloc()` can
1. `new` is an operator, while `malloc()` is a function.
   /home/akong_yin/my-repos/the-nitty-gritty/c-cpp/intersection/11_posix-api
