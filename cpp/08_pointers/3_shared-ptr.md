# Pointer Series: Ep. 3 - `std::shared_ptr`

- `std::shared_ptr` is a smart pointer that retains shared ownership of an object through a pointer. Several shared_ptr
  objects may own the same object [[1](https://en.cppreference.com/w/cpp/memory/shared_ptr)] through sharing a resource
  counter[[2](https://courses.cs.washington.edu/courses/cse390c/24wi/lectures/smart.pptx)]:
    - Constructors will create the counter
    - Copy constructor and operator= will increment the counter
    - Destructor will decrement the counter

- The ownership of an object can only be shared with another shared_ptr by copy constructing or copy assigning its value
  to another shared_ptr. Constructing a new shared_ptr using the raw underlying pointer owned by another shared_ptr
  leads to undefined behavior [[3](https://en.cppreference.com/w/cpp/memory/shared_ptr)].
-
- shared_ptr objects replicate a limited pointer functionality by providing access to the object they point to through
  operators * and ->. For safety reasons, they do not support pointer
  arithmetics [[4](https://cplusplus.com/reference/memory/shared_ptr/)].