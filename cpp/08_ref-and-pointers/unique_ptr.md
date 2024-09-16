# Smart pointer

- Smart pointers is a much-hyped function in "modern" C++. They are included
  in the Standard Library and they are used to help ensure that programs are
  free of memory and resource leaks and are exception-safe.

- In C++, a smart pointer is implemented as a template class that mimics, by
  means of operator overloading, the behaviors of a traditional (raw) pointer,
  (e.g. dereferencing, assignment) while providing additional memory management
  features.

- Practically, the implementation of smart pointers is based on the idea of
  [RAII](../01_raii-and-the-rule-of-five):
  - In a smart pointer's constructor, memory is allocated; in a smart
    pointer's destructor, memory is deallocated.
  - So when a smart pointer object goes out of scope, following the RAII
    principle, the smart pointer's destructor is called and memory gets
    automatically released.

## `std::unique_ptr<>`

- This is the most basic type of smart pointer. As its name suggests, A
  `unique_ptr` does not share its pointer--meaning that its internal data buffer
  cannot be copied to another `unique_ptr`, passed by value to a function, or
  used in any C++ Standard Library algorithm that requires copies to be made.

  - A `unique_ptr` can be moved--this means that the ownership of
    the memory resource is transferred to another `unique_ptr` and the
    original `unique_ptr` no longer owns it:

  ```C++
  std::unique_ptr<int> smart_int_arr2 = std::move(smart_int_arr);
  ```

- As always, the first hello world program looks sane and simple, but...things
  becomes more complicated, when we try to incoporate existing C code into C++.

  - Consider the following example, we get (and own) a dynamic integer
    array from an existing C function. Current implementation means we need to
    manually `free()` it after use. We want to convert the use of raw pointers
    to smart pointers. We do it by handing over the ownership of a raw pointer
    to a smart pointer and forget the raw pointer, and then memory management
    will be automatically handled by the smart pointer's destructor.:

  ```C++
  int* dynamic_int_arr = (int*)malloc(sizeof(int) * arr_size);
  // In reality, malloc() should be some deep-rooted C functions,
  // probably in a compiled so file. Here we simplify the scenario by
  // directly using a malloc()
  for (int i = 0; i < arr_size; i++) {
      dynamic_int_arr[i] = i;
  }
  std::unique_ptr<int[]> smart_int_ptr(dynamic_int_arr);
  ```

  - What's wrong? Think about it: for raw pointers, we can either `new` it
    or `malloc()` it with some memory. Following the principle of RAII, an
    object's desctructor will be called when it goes out of scope. When a
    smart pointer's destructor is called, how does it know that it should
    call `delete` or `free()`? No, it doesn't ¯\\\_(ツ)\_\/¯ and it simply
    calls `delete`/`delete[]`.
  - Calling `delete` forcefully against a `malloc()`ed pointer results in
    undefined behaviors.
  - To properly handle this, C++ introduces yet another layer of complexity--
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

  - It is also possible to make it a oneliner:

  ```C++
  std::unique_ptr<int[], decltype([](void *p){std::free(p);})>
  ```

## `unique_ptr` as argument or return value

- As `unique_ptr` can not be copied, how can we pass it to a function as an
  argument? There are at least three feasible ways, we pass it:

  - as a raw pointer,
  - by reference, or
  - use `std::move()`:

  ```C++
  void callee_func_raw_ptr(int *arg) {}
  void callee_func_ref(unique_ptr<int> &arg) {}
  void callee_func_move(unique_ptr<int> arg) {}

  unique_ptr<int> x(new int(0));
  *x = 45;
  callee_func_ref(x);
  callee_func_raw_ptr(x.get());
  callee_func_move(move(x));
  ```

- It is legal to return a `unique_ptr` like the following:

```C++
unique_ptr<int> get_ptr() {
  unique_ptr<int> x(new int(31415));
  return x;
}
```

- The exact mechanism of this may differ. The compiler may either apply copy/move
  elision (a.k.a. RVO) or, without elisions, `move()` the ownership of the returning
  `unique_ptr` to the new `unique_ptr` assigned with the return value.

## Is `unique_ptr` a zero-cost wrapper on top of raw pointer?

- TL;NR: No.

- When we say we "use" a pointer, we usually mean by using either `operator*`
  or `operator->`. In this regard, `unique_ptr` is indeed a zero-cost wrapper.
  It is trivial for a compiler to optimize the abstraction away and make
  the smart pointer work as if it is a raw pointer.

- The hidden cost arises when we want to enjoy the benefit of RAII (i.e., when
  we initialize or destory a `std::unique_ptr`).

  - This cost will be more pronounced if we use RAII frequently according to
    the following example from seminar
    [CppCon 2019 - Chandler Carruth "There Are No Zero-cost Abstractions"](https://www.youtube.com/watch?v=rHIkrotSwcc):

  ```C++
  void bar(int* ptr) noexcept;

  void baz(unique_ptr<int>&& ptr) noexcept;

  void foo(unique_ptr<int>&& ptr) {
    if (*ptr > 42) {
      bar(ptr.get());
      *ptr = 42;
    }
    baz(std::move(ptr));
  }
  ```

  - If we do it this way, the compiler could emit unexpected instructions
    which may need to have extra memory hit.

- Another hidden cost, according to
  [this post](https://www.thecodedmessage.com/posts/cpp-move/)
  is the use of move semantics.

  - When we `std::move()` the resource managed by one `unique_ptr`, (`a`) to
    anthoer `unique_ptr`, (`b`), we need to set the internal raw pointer of
    `a` to something like `nullptr`.
  - Also, `a`'s destructor will be called anyway, even if we set `a`'s
    internal raw pointer to `nullptr`. Calling the "almost empty" destructor
    incurres another cost.

- While the "normal use" of a `unique_ptr` should be exactly the same as a raw
  pointer, one has to be careful when moving/passing around a `unique_ptr`
  frequently--it could be significantly worse than its hand-written C version.

## References

- [Microsoft - Smart pointers (Modern C++)](https://learn.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=msvc-170)
- [Wikipedia - Smart pointer](https://en.wikipedia.org/wiki/Smart_pointer)
- [CPP Reference - Smart pointers](https://en.cppreference.com/book/intro/smart_pointers)
