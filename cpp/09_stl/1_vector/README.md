# Standard Template Library (STL)

- The STL is one of my favorite C++ features. Class/reference/exception/etc are useful but not too troublesome to
  replicate in C. But speak of STL, it could be really tricky for application developers to reinvent the wheels prepared
  by the STL, let alone making the wheels robust and efficient.

  - Well perhaps credit should be given to templates instead, as it is what makes STL possible.

- This sub-project does not intend to cover all the details of STL and it will contain a few possibly confusing
  points only.

## Vector

- Vectors are the same as dynamic arrays with the ability to resize itself automatically when an element is inserted
  or deleted, with their storage being handled automatically by the container.

  - Note that although the size of a vector is automatically handled and can be dynamically adjusted, internally
    it still uses a C array, instead of a linked list, etc.
  - Vector elements are placed in contiguous storage.

- To get the address of a vector's interal storage, there are at least five methods: `&vec[0]`, `vec.data()`,
  `vec.begin()`, `&vec.front()`, `&vec.at(0)`.

- As long as a `vec` is large enough, `memcpy()` works the same as in c: `memcpy(vec.data(), arr, sizeof(arr));`.

- To add a new element (by value) at the end of a vector, use `vec.push_back()`. As a vector typically reserves some
  storage space at the end , if not being use too much, `push_back()` is usually an `O(1)` operation. However,
  if a `push_back()` call requires an internal C array size that is greater than its currently "capacity", the vector
  may need to re-`malloc()` to accommodate.

- To add a new element at the beginning of a vector with `vec.insert(vec.begin(), value);` is more
  expensive and it is always an `O(n)` operation. Because each existing element will be moved by one index internally
  to make the room.
  - If we know in advance that many elements need to be `insert()`ed at the beginning, it could be more efficient
    if we define another vector with enough space and copy new and existing elements to it.
