# Template

* Templates are a very powerful C++ feature that allows functions and classes
to operate with generic types.
    * It behaves similarly to C#/Java's generics (But underlyingly they are different).
    * Some people argue that it is [more important than inheritance](https://youtu.be/HqsEHG0QJXU?t=133)
    * Personally, it seems to be even more important than class/object support--
        * In C, we can implement the OOP principle by using structs and
          organizing/naming functions properly--it does impose some extra
          mental burden but is still manageable.
        * But we don't have a way to achieve what templates can do without
          using ugly and bug-prone Marcos, which is rarely practical beyond
          a hello world proof-of-concept.

* The purpose of templates in C++ is rather straightforward. In the bad old days (i.e., in C), re-using function
definitions for different data types is a nightmare. To awkwardly achieve the same goal, we can do it:
  * The copy-n-paste way:
    ```C
    int my_max_int(int a, int b) {
      return (a > b) ? a : b;
    }
    double my_max_dbl(double a, double b) {
      return (a > b) ? a : b;
    }
    char* my_max_chr(char* a, char* b) {
      return (a > b) ? a : b;
    }
    ```
  * Or, the "automatic" way:
    ```C
    #define BUILD_COMPARE(TYPE) \
    int cmp_ ## TYPE(const void* va, const void* vb) \
    { \
    TYPE const* pa = static_cast<TYPE const*>(va); \
    TYPE const* pb = static_cast<TYPE const*>(vb); \
    \
    if (*pa < *pb) return -1; \
    else if (*pa == *pb) return 0; \
    else return 1; \
    }
    BUILD_COMPARE(float)
    BUILD_COMPARE(double)
    ```

* The issue becomes glaring if we want to implement something generic, say, Quick Sort for different data types.

* In C++, it is done with templates as shown [here](./1_hello-world.cpp). At first glance, it is plain and simple--
`template<typename T>` is used to designate a "to-be-specified" data type. When a caller calls the function, the
type is specified and the function runs smoothly.
  * Well as always, the devil is in the details. The C++'s way is more than meets the eye...

## [1_hello-world.cpp](./1_hello-world.cpp):

* It is a prime example of C++'s ["as-if" rule](https://en.cppreference.com/w/cpp/language/as_if).
  * Theoretically, it means that C++ allows any and all code transformations that do not change the observable
  behavior of the program. 

* If we examine the assembly code of the main function:
  ```assembly
  max_int = my_max(1, 3);
  cout << max_int << endl;
  11f2:	be 03 00 00 00       	mov    esi,0x3
  11f7:	48 8d 3d c2 2e 00 00 	lea    rdi,[rip+0x2ec2]        # 40c0 <_ZSt4cout@@GLIBCXX_3.4>
  ```
  One can observe that there is no function calls at all! All the machine code does is put `0x3` to the
  `esi` register and call `cout` to print it.
  * Reason? Because `g++` detects that `max_int = my_max(1, 3);` is essentially a const, `3`, then why call the
  `my_max()` function in runtime again and again? Let's just print `3`. Brilliant!

* To summarize, in `1_hello-world.cpp`, we demonstrated how templates are written in C++'s source code--but after
`g++`'s optimization, the `my_max()` function and all the templates stuff are nowhere to be found on the
machine code level.
  * This is also one of the key reasons why C++ can be almost as fast as C--while it adds a lot of useful abstraction
  on top of C, at the bare metal level, all the abstraction goes away and the machine code should, hopefully, be
  as good as C (or, as good as "as-if" such abstraction never has existed).