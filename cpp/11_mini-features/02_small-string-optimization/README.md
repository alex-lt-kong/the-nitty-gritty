# Small-string optimization

* Generally speaking, C++'s `std::string` is a `std::vector<char>` in disguise
with a few useful features[[1]].
    * However, if a string is known to be very short, we may store it on stack,
    instead of on heap, to increase performance.

* There are a few reasons why storing on stack can be faster than on heap,
including:
    1. To get memory from heap, one needs to call some form of `malloc()`,
    which is not a cheap operation.
    1. Stack data are physically close to each other, making the caching
    mechanism, notably the cache line mechanism, much more efficient.
    1. The address/offset of variables on stack is more likely to be calculated
    at compile time. Comparatively, the address/offset of variables on heap
    is more likely to be only calculated at runtime.

* On x64 systems, a `std::string` variable is roughly a `struct` of three
8-bytes variables:
    1. a `char* ptr` variable storing the location of real `char`s of the 
    string.
    1. a `size_t size` variable storing the current size of the string.
    1. a `size_t capacity` variable storing the allocated memory of the string.

* The basic idea of small-string optimization (SSO) is rather simple--if we
know that `strlen()` of a string is less than 24, then we can directly store
it on stack without imposing any extra burden to it (for x32 system the
threshold is 12).
    * Maximum `strlen()` is limited to 23/11 as we need to add a `\0` to the
    end of it.

* Also note that SSO is a compiler-specific function--while major compilers
all implement it one way or another, a standard-compliant compiler may choose
not to implement at all.

* In the assembly generated for `ssoNaiveDemo()`, we can see that the string
`foobar` is directly stored on stack:
    ```asm
    2:	b8 61 72 00 00       	mov    eax,0x7261
    ...
    18:	66 89 44 24 14       	mov    WORD PTR [rsp+0x14],ax
    2e:	c7 44 24 10 66 6f 6f 	mov    DWORD PTR [rsp+0x10],0x626f6f66
    ...
    ```

## Reference

1. [CPP Optimizations diary  - Small String Optimizations][1]
1. [C++ and more - The Small String Optimization][2]

[1]: https://cpp-optimizations.netlify.app/small_strings/ "CPP Optimizations diary  - Small String Optimizations"
[2]: https://blogs.msmvps.com/gdicanio/2016/11/17/the-small-string-optimization/ "C++ and more - The Small String Optimization"