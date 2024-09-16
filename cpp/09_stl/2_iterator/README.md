# Iterators

- Iterators are one of the four pillars of the Standard Template Library or STL in C++. (the remaining three being algorithms, containers and functors (aka function objects))

- An iterator is a pointer-like object representing an element's position in a container. It is used to iterate over elements in a container.

  - Concretely, an iterator is a simple class that provides a bunch of operators: increment ++, dereference \* and few others which make it very similar to a pointer and the arithmetic operations you can perform on it.

    ![a vector's iterator, from https://www.programiz.com/sites/tutorial2program/files/vector-iterator.png](./assets/vector-iterator.png)

- Modern C++ defines six types, with later types build upon eariler types:

  1. Input Iterator: Can scan the container forward only once, can't change the value it points to (read-only);
  2. Output Iterator: Can scan the container forward only once, can't read the value it points to (write-only);
  3. Forward Iterator: Can scan the container forward multiple times, can read and write the value it points to;
  4. Bidirectional: Iterator Same as previous one but can scan the container back and forth;
  5. Random Access: Iterator Same as previous one but can access the container also non-sequentially (i.e. by jumping around);
  6. Contiguous Iterator: Same as previous one, with the addition that logically adjacent elements are also physically adjacent in memory.

![iterators' hierarchy, from https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp](./assets/iterator-hierarchy.png)
