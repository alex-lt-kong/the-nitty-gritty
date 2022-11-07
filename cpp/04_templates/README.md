# Templates

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

* The issue becomes glaring if we want to implement something generic but complicated, say, Quick Sort
  for different data types.
  * Well one can argue that quick sort isn't really complicated ¯\\_(ツ)_/¯

* In C++, it is done with templates as shown in [1_hello-world.cpp](./1_hello-world.cpp).
At first glance, it is plain and simple--`template<typename T>` is used to designate a "to-be-specified"
data type. When a caller calls the function, the type is specified and the function runs smoothly.
  * Well as always, the devil is in the details. The C++'s way is more than meets the eye...
  * Here we prepared a few demo programs to reveal how templates work.

## [1_hello-world.cpp](./1_hello-world.cpp):

* It is a prime example of C++'s ["as-if" rule](https://en.cppreference.com/w/cpp/language/as_if).
  * Theoretically, it means that C++ allows any and all code transformations that do not change the observable
  behavior of the program.
  * Simply put, compilers are allowed to do whatever they want, as long as the program works "as-if" all C++ source
  code is translated to machine code authentically and naively.

* If we examine the assembly code of the main function:
  ```assembly
  max_int = my_max(1, 3);
  cout << max_int << endl;
  11f2:	be 03 00 00 00       	mov    esi,0x3
  ...
  121b:	e8 f0 fe ff ff       	call   1110 <std::ostream::operator<<(int)@plt>
  ```
  One can observe that there is no function calls at all! All the machine code does is put `0x3` to the
  `esi` register and call `cout` to print it.
  * Reason? Because `g++` detects that `max_int = my_max(1, 3);` is essentially a const, `3`, then why call the
  `my_max()` function in runtime again and again? Let's just print `3`. Brilliant!

* To summarize, in `1_hello-world.cpp`, we demonstrated how templates are written in C++'s source code--but after
`g++`'s optimization, the `my_max()` function and all the templates stuff are nowhere to be found at the
machine code level.
  * This is also one of the key reasons why C++ can be almost as fast as C--while it adds a lot of useful abstraction
  on top of C, at the bare metal level, all the abstraction goes away and the machine code should, hopefully, be
  as good as C (or, as good as "as-if" such abstraction never has existed).
  * This is also one reasons why C++ is usually slightly slower then C--some of its abstraction is
  difficult or impossible to be optmized away, dragging the performance down a bit.

## [2_dynamic-input.cpp](./2_dynamic-input.cpp):

* To overcome the ["as-if"](https://en.cppreference.com/w/cpp/language/as_if) trap, in this program
we make input dynamic by using the `rand()` function.

* The resultant assembly code is closer to what we expect:
  ```assembly
      int a_int = rand(), b_int = rand();
      1104:	e8 37 ff ff ff       	call   1040 <rand@plt>
      1109:	89 c3                	mov    ebx,eax
      110b:	e8 30 ff ff ff       	call   1040 <rand@plt>
      max_int = my_max(a_int, b_int);
      cout << max_int << endl;
      1110:	48 8d 3d 69 2f 00 00 	lea    rdi,[rip+0x2f69]        # 4080 <std::cout@@GLIBCXX_3.4>
      1117:	39 c3                	cmp    ebx,eax
      1119:	0f 4d c3             	cmovge eax,ebx
      111c:	89 c6                	mov    esi,eax
      111e:	e8 ad ff ff ff       	call   10d0 <std::ostream::operator<<(int)@plt>
  ```
  * It is good to know that we finally have a `cmp` instruction:

    0. It `call`s `rand()` at `1040` and the return value is stored in `eax`.
    0. It `mov`s value in `eax` to `ebx` to make place for the 2nd `rand()` call.
    0. It `call`s `rand()` again, storing the value in `eax`
    0. It `cmp`s `ebx` and `eax` and sets [`FLAGS` register](https://en.wikipedia.org/wiki/FLAGS_register) accordingly.
    0. It `cmovge`s, i.e., it conditionally moves value from source register to destination register
    by checking the value of a `FLAGS` register (probably the SF register)
    0. It `call`s `<<` at `10d0` to print the result in `esi` to stdout.
    0. The above operations are doing essentially this:
    ```C
    int a_int = rand(), b_int = rand();
    int max_int = a > b ? a : b;
    cout << max_int << endl;
    ```
  * But where is the function call?
    * The `as-if rule` is applied again--the function has been inlined--`g++` may think that it is not worth it
    to call the one-liner function given the extra work needed by making a function call. Instead, it simply
    copies and pastes the function body directly into the `main()` function.

* The `double` version of `my_max()` shows something similar:
  ```assembly
    double max_dbl;
    double a_dbl = (double)rand() / rand();
    112b:	e8 10 ff ff ff       	call   1040 <rand@plt>
    1130:	89 c3                	mov    ebx,eax
    1132:	e8 09 ff ff ff       	call   1040 <rand@plt>
    1137:	66 0f ef c0          	pxor   xmm0,xmm0
    113b:	66 0f ef c9          	pxor   xmm1,xmm1
    113f:	f2 0f 2a c8          	cvtsi2sd xmm1,eax
    1143:	f2 0f 2a c3          	cvtsi2sd xmm0,ebx
    1147:	f2 0f 5e c1          	divsd  xmm0,xmm1
    114b:	f2 0f 11 44 24 08    	movsd  QWORD PTR [rsp+0x8],xmm0
    double b_dbl = (double)rand() / rand();
    1151:	e8 ea fe ff ff       	call   1040 <rand@plt>
    1156:	89 c3                	mov    ebx,eax
    1158:	e8 e3 fe ff ff       	call   1040 <rand@plt>
    115d:	66 0f ef c9          	pxor   xmm1,xmm1
    1161:	66 0f ef d2          	pxor   xmm2,xmm2
    return a > b ? a : b;
    1165:	f2 0f 10 44 24 08    	movsd  xmm0,QWORD PTR [rsp+0x8]
    double b_dbl = (double)rand() / rand();
    116b:	f2 0f 2a d0          	cvtsi2sd xmm2,eax
       *  These functions use the stream's current locale (specifically, the
       *  @c num_get facet) to perform numeric formatting.
      */
      __ostream_type&
      operator<<(double __f)
      { return _M_insert(__f); }
    116f:	48 8d 3d 0a 2f 00 00 	lea    rdi,[rip+0x2f0a]        # 4080 <std::cout@@GLIBCXX_3.4>
    1176:	f2 0f 2a cb          	cvtsi2sd xmm1,ebx
    117a:	f2 0f 5e ca          	divsd  xmm1,xmm2
    return a > b ? a : b;
    117e:	f2 0f 5f c1          	maxsd  xmm0,xmm1
    1182:	e8 39 ff ff ff       	call   10c0 <std::ostream& std::ostream::_M_insert<double>(double)@plt>
    1187:	48 89 c7             	mov    rdi,rax
	return __pf(*this);
    118a:	e8 31 01 00 00       	call   12c0 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>
    max_dbl = my_max(a_dbl, b_dbl);
    cout << max_dbl << endl;
    return 0;
    118f:	48 83 c4 10          	add    rsp,0x10
    1193:	31 c0                	xor    eax,eax
    1195:	5b                   	pop    rbx
    1196:	c3                   	ret    
    1197:	66 0f 1f 84 00 00 00 	nop    WORD PTR [rax+rax*1+0x0]
    119e:	00 00 
  ```
  * While the `double` version is much more complicated, it essentailly does the same:
    * After generating two random doubles, at `117e`, `xmm0` stores `a_dbl` while `xmm1` stores `b_dbl`.
    * `call`s something at `10c0`, which is believed to be the `cout` function.