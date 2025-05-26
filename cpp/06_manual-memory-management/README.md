# Manual memory management

## Common ways and when to use them

- `new`/`new[]` and `delete`/`delete[]`: Most common ones, use `new`/`new[]` to
  allocate and `delete`/`delete[]` deallocate.
-
- `malloc()` and `free()`
    - allocation/deallocation defined in C standard library.
    - They are not strongly typed and they dont call constructor/destructor by
      default. Need to use placement new/`std::construct_at()` to initialize the
      memory being allocated and call destructor manually:

    ```C++
    auto *raw_ptr = static_cast<MyClass *>(malloc(sizeof(MyClass)));
    const auto ptr = new (raw_ptr) MyClass();
    ptr->~MyClass();
    free(ptr);
    ```

- `std::allocator<T>`
    - Similar to `malloc()`/`free()` but strongly-typed.
    - Until C++17, `std::allocator<T>` defines `construct()` and `destroy()`,
      but since C++20 they are not replaced by `std::construct_at()` and
      `std::destory_at()`, making them look more similar to `malloc()`/
      `free()` <sup>[[std::allocator<T>::construct](https://en.cppreference.com/w/cpp/memory/allocator/construct)]</sup>

- `std::allocator<T>` is not meant to replace `new`/`new[]` and `delete`/
  `delete[]`, they are used when you want to separate allocation and
  construction into two steps (and similarly to separate destruction and
  deallocation into two
  steps). <sup>[[What's the advantage of using std::allocator instead of new in C++?](https://stackoverflow.com/questions/31358804/whats-the-advantage-of-using-stdallocator-instead-of-new-in-c)]</sup>
  For example:

    ```C++
    std::vector<std::string> vec;
    v.reserve(4);        // allocate, but dont construct
    v.push_back("Hello world!\n");
    v.push_back("0xdeedbeaf");
    v.clear();           // destruct, but dont deallocate
    ```

## `placement new` and `std::construct_at`/`std::destroy_at`