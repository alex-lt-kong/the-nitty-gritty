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
    to call the one-liner function given the functional-call overhead. Instead, it simply copies and pastes the
    function body directly into the `main()` function.

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

## [3_noinline.cpp](./3_noinline.cpp)

* There are a few ways that we can choose to prevent our templates functions from being inlined. Here we pick the
`g++`-specific approach--adding the attribute `__attribute__((noinline))` to the templates function `my_max()`
according to gcc's [official documentation](https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#Common-Function-Attributes).

* Now let's take a look at the assembly code generated for the `int` function call.
  ```assembly
    int max_int;
    int a_int = rand(), b_int = rand();
    1105:	e8 36 ff ff ff       	call   1040 <rand@plt>
    110a:	89 c5                	mov    ebp,eax
    110c:	e8 2f ff ff ff       	call   1040 <rand@plt>
    max_int = my_max(a_int, b_int);
    1111:	89 ef                	mov    edi,ebp
    int a_int = rand(), b_int = rand();
    1113:	89 c6                	mov    esi,eax
    max_int = my_max(a_int, b_int);
    1115:	e8 16 02 00 00       	call   1330 <int my_max<int>(int, int)>

    ...

    0000000000001330 <int my_max<int>(int, int)>:
    return a > b ? a : b;
    1330:	39 f7                	cmp    edi,esi
    1332:	89 f0                	mov    eax,esi
    1334:	0f 4d c7             	cmovge eax,edi
  }
    1337:	c3                   	ret    
    1338:	0f 1f 84 00 00 00 00 	nop    DWORD PTR [rax+rax*1+0x0]
    133f:	00 
  ```
  * Good, the function call is finally there. One can observe that opcodes/operands are pretty similar to those
  generated for [2_dynamic-input.cpp](./2_dynamic-input.cpp), just a `call` instruction is added.

* How about the double version of the call?
  ```assembly
    double max_dbl;
    double a_dbl = (double)rand() / rand();
    1130:	e8 0b ff ff ff       	call   1040 <rand@plt>
    1135:	89 c3                	mov    ebx,eax
    1137:	e8 04 ff ff ff       	call   1040 <rand@plt>
    113c:	66 0f ef c0          	pxor   xmm0,xmm0
    1140:	66 0f ef c9          	pxor   xmm1,xmm1
    1144:	f2 0f 2a c8          	cvtsi2sd xmm1,eax
    1148:	f2 0f 2a c3          	cvtsi2sd xmm0,ebx
    114c:	f2 0f 5e c1          	divsd  xmm0,xmm1
    1150:	f2 0f 11 44 24 08    	movsd  QWORD PTR [rsp+0x8],xmm0
    double b_dbl = (double)rand() / rand();
    1156:	e8 e5 fe ff ff       	call   1040 <rand@plt>
    115b:	89 c3                	mov    ebx,eax
    115d:	e8 de fe ff ff       	call   1040 <rand@plt>
    1162:	66 0f ef c9          	pxor   xmm1,xmm1
    1166:	66 0f ef d2          	pxor   xmm2,xmm2
    max_dbl = my_max(a_dbl, b_dbl);
    116a:	f2 0f 10 44 24 08    	movsd  xmm0,QWORD PTR [rsp+0x8]
    double b_dbl = (double)rand() / rand();
    1170:	f2 0f 2a d0          	cvtsi2sd xmm2,eax
    1174:	f2 0f 2a cb          	cvtsi2sd xmm1,ebx
    max_dbl = my_max(a_dbl, b_dbl);
    1178:	f2 0f 5e ca          	divsd  xmm1,xmm2
    117c:	e8 bf 01 00 00       	call   1340 <double my_max<double>(double, double)>

    ...

    0000000000001340 <double my_max<double>(double, double)>:
    return a > b ? a : b;
    1340:	f2 0f 5f c1          	maxsd  xmm0,xmm1
  }
    1344:	c3                   	ret    
    1345:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
    134c:	00 00 00 
    134f:	90                   	nop
  ```
  * The function call is there as well. Note that the signature of the call is
  `<double my_max<double>(double, double)>`, i.e., at assembly code level,
  the templates function are gone--instead, a proper `double` function is created.

* We finally come to the first important point about templates: it is
[a parametrized description of a family of classes/functions](https://cppcon.digital-medium.co.uk/wp-content/uploads/2021/11/back_to_basics_templates_part_1__bob_steagall__cppcon_2021.pdf),
not a class/function per se. To be specific, it means:
  * templates classes/functions are not directly compiled to machine code as
  one single "versitle" or "generic" function/class that can magically take
  a few different parameters. Instead, the compiler generates a series of
  concrete functions/classes, plugging in concrete data types, such as
  `int`, `double` as needed.
    * Note that "as needed" means if a version is never used, it will
    not be generated. In our example, only `int` and `double` versions
    are generated, all other versions, such as `string`, are not generated.
  * This paradigm is called
  [generic programming](https://en.wikipedia.org/wiki/Generic_programming).
  * It works a bit similar to Marco in C in the sense that both can
  "generate" code to be compiled. But templates are more than just
  being prettier, it offers concrete benefits such as double-increment
  protection and type check. Take this simple marco as an example:
    ```C
      #define max(a,b) (a) > (b) ? (a) : (b)
    ```
    At first glance, it is as capable as templates--all variables are
    properly parenthesized so we avoid any unexpected "escape" or
    "truncation". However, it could still behave erroneously if some handpicked
    arguments are "passed" to it:
    ```C
    int a = 1, b = 0;
    max(a++, b);      // a is incremented twice, (because a > b is true)
    max(a++, b+10);   // a is incremented once,  (becuase a > b is false)
    max(a, "Hello");  // comparing ints and ptrs, doesn't make sense but may compile
    ```
    * According to [this presentation](https://cppcon.digital-medium.co.uk/wp-content/uploads/2021/11/back_to_basics_templates_part_1__bob_steagall__cppcon_2021.pdf), Marco and templates also differ in
    another aspect: Marcos are parsed in the pre-processing stage and templates are parsed in the compilation stage
    of the process that translates source code to machine code.
  * C++'s templates also differ greatly from "generics" in Java/C#. Bjarne Stroustrup
  [argues that](https://www.stroustrup.com/bs_faq2.html#generics) generics are
  primarily syntactic sugar for abstract classes; that is, with generics, you program against precisely
  defined interfaces and typically pay the cost of virtual function calls and/or dynamic casts to use arguments. 
  * By taking the current approach, C++ is able to offer a similar level of flexibility while not compromising
  on performance
    * As all templates abstraction goes away at assembly level, C++ code with templates should be as performant
    as its C counterpart.
    * A less desirable result of the approach is late detection of errors and horrendously bad error messages.

## [4_edge-cases.cpp](./4_edge-cases.cpp)

* From users' (i.e., C++ programmers) point of view, the general use of
templates should be straightforward. However, edge cases are everywhere
and C++ has many detailed rules to make them well-defined.
  * This kind of pursuit of completeness, as far as I am concerned,
  is also one of the main reasons why C++ is usually considered more
  complicated than other high-level languages.

* Explicit specification of typename
  * Consider the following function template definition:
  ```C++  
  template<typename T>
  __attribute__((noinline)) T my_max(T a, T b) {
      return a > b ? a : b;
  }
  ```
  * What if we pass an `int` to a and a `double` as b?
  ```C++
  int max_int;
  int a_int = rand();
  double b_dbl = (double)rand() / rand();
  // max_dbl = my_max(a_dbl, b_int); won't compile, C++ doesn't know which type should be used
  max_dbl = my_max<double>(a_dbl, b_int);
  ```
  * No, the code won't compile. While it is trival for human beings to note we mostly want to convert `int` to
  `double` to make it work. C++ compilers won't do this for us--unless we make it explicit.

* Partially available templates operations
  * Consider the following class template definition:
  ```C++
  template<typename T>
  class SimpleClass {
  private:
      T first;
      T second;
  public:
      SimpleClass(T first, T second) {
          this->first = first;
          this->second = second;
      }
      ~SimpleClass() {}    
      T getSum() {
          return first + second;
      }
      T getProduct() {
          return first * second;
      }
  };
  ```
  * instantiating an `int` version of the class template and invoke both methods are fine since both
  `+` and `*` are defined for integers:
  ```C++
  SimpleClass<int> MyIntClass(1, 2);
  MyIntClass.getSum();
  MyIntClass.getProduct();
  ```
  * But only `+` operation is defined between two string while `*` isn't,
  can we instantiate a `string` version of the class template?
  ```C++
  SimpleClass<string> MyStrClass(str1, str2);
  MyStrClass.getSum();
  // MyStrClass.getProduct()； won't work, as string * string is not defined.
  ```
    * The answer is affirmative--as long as we don't invoke the unsupported methods.
  * This behavior is consistent with what we observed in [3_noinline.cpp](./3_noinline.cpp). The idea is that only
  the versions actually being used are generated--so as long as we don't touch the unsupported methods, we 
  are still good.

## [5_trait.cpp](./5_trait.cpp)

* Sometimes using templates once is not enough to handle a few more flexible
cases. Let's say, we want to write a function that takes a numeric parameter
`val` and it wants to check if `val` is greater than the half of the maximum
possible value of its type.
  * For example, a function for `int` in C is like this:
    ```C
    bool is_greater_than_half_max_int(int val) {
      return (val > INT_MAX / 2);
    }
    ```
    and for `float`, a separate function should be like this:
    ```C
    bool is_greater_than_half_max_float(float val) {
      return (val > FLOAT_MAX / 2);
    }
    ```


* In C++, we would like to take advantage of templates so that we only need to
write one function template to handle all possible numeric types, something
like this:
  ```C++
  template <typename T>
  bool greater_than_half_max_with_trait(T val) {
      return (val > T_MAX / 2);
  }
  ```
  * But wait, what will be T_MAX here? In C, `INT_MAX` and `FLOAT_MAX` are
  just marcos that expand to 2147483647, 3.40282346638528859812e+38F, etc,
  how can we have `T_MAX`?

* Trait comes to our rescue!
  ```C++
  template <typename T>
  bool greater_than_half_max_with_trait(T val) {
      return (val > numeric_limits<T>::max() / 2);
  }
  ```
  * In C++'s standard library a `numeric_limits` template is prepared, we just
  pass `T` to it and it will "return" the `T_MAX` accordingly, at compile time.
  * `numeric_limits` is a templates class, for sure. But for this particular
  case, we also call it a "type trait".

* Long story short, trait is a kind of application of templates that helps
us handle diversity of different types that we don't care and provides the
common features that we do care.

* One thing that is worth pointing out is that traits is only capable of
hiding the details we dont want to touch, but it can't get rid of the
details altogether.
  * Using `numeric_limits<T>` as an example, to implement our
  `greater_than_half_max_with_trait()` with it, we can create one template
  function instead of a bunch of concrete functions, great!
  * However, in STL's `limits` file, the implementer has to prepare quite a 
  few concrete implementations, such as `numeric_limits<int>`, 
  `numeric_limits<double>`, `numeric_limits<uint32_t>`, etc.
  * The benefit is only that as language users (i.e., developers), we can
  get away from this trouble, it doesn't mean the trouble magically
  disappears--just we shift the burden to the implementer of STL.

## Templates in shared object files

* It is not uncommon for people to pre-compile source code into `.so` files
and distribute them along with a `.h` file for downstream users. How can we
do this for templates functions/classes?
  * No, this is impossible🤦

* The reason lies in how templates are designed.
  * Note that template functions/classes are not functions/classes
  themselves, the concrete functions/classes get generated after the
  `typename T` is fixed.
  * When a source code file is compiled into a `.so` file, no function calls
  are known to the compiler and thus there is no way for compilers to
  generate concrete functions/classes out of nothing.

* But if this is the case, then how is the STL and other templates-enabled
libraries are distributed?
  * The common way is plain and simple--all the templates are defined in
  header files only, so no `.c`, `.cpp` or whatever source code files are needed.
  * We an STL or other functions are called, only the called version will
  be instantiated.

* There is one exception though.
  * If we know that only a few variants of the templates functions/classes
  will be used, we can instantiate those versions explicity
  [this way](https://stackoverflow.com/a/1022676/19634193).
  * These concrete functions/classes can be compiled into `.so` files. But
  this way, they fall back to a series of common functions/classes and the
  concept and implementation of the "template" concept will be totally gone.

## Important notes

* As demonstrated above, a seemingly stragitforward feature can get
really sophisticated in C++.
* But let's not forget Bjarne Stroustrup's famous quote that
[within C++, there is a much smaller and cleaner language struggling to get out](https://www.stroustrup.com/quotes.html).
* The quote can be interpreted in a few different ways, but my take is: if
we stick to templates in its common form, and we just don't touch those
archaic and cryptic (although still well-defined) cases, templates should
"just work".
* On the other hand, if we try to go down the rabbit hole, the journey
will be much more diffcult and mostly it doesn't really help solving
practical engineering problems.

## Reference

1. [Lei Mao - C++ Traits][1]
1. [A quick primer on type traits in modern C++][2]

[1]: https://leimao.github.io/blog/CPP-Traits/ "Lei Mao - C++ Traits"
[2]: https://www.internalpointers.com/post/quick-primer-type-traits-modern-cpp "A quick primer on type traits in modern C++"