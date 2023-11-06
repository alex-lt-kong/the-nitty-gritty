# Undefined behaviors

- While the general idea of undefined behavior is not difficult to understand,
  the exact wording may vary. [C11 standard][1] defines undefined behavior as
  follows:
  > undefined behavior is otherwise indicated in this International Standard by
  > the words ‘‘undefined behavior’’ or by the omission of any explicit
  > definition of behavior.
- In the C community, undefined behavior may be humorously referred to as
  "nasal demons", after a comp.std.c post that explained undefined behavior as
  allowing the compiler to do anything it chooses, even "to make demons
  fly out of your nose".
- It is common for programmers, even experienced ones, to rely on undefined
  behavior either by mistake, or simply because they are not well-versed in
  the rules of the language that can span hundreds of pages. This can result
  in bugs that are exposed when a different compiler or lead to security
  vulnerabilities in software.

- But if undefined behaviors are so bad, why don't we just define them?
  1.  Documenting an operation as undefined behavior allows compilers to assume
      that this operation will never happen in a conforming program. This gives
      the compiler more information about the code and this information can lead to
      more optimization opportunities.
  1.  For some ambiguious cases, different platforms/hardware designs
      could strongly favor different outcomes, if C standard mandates one
      outcome, it may impose a significant performance penalty on some
      platforms.
  1.  Simply put, the existence of undefined behaviors makes C fast!

## Some common undefined behaviors

### [1. Format specifier without argument](./01_format-specifier-without-argument/)

- Source: para. 2 of section of 7.16.1.1 of [C11][1]:

  > If there is no actual next argument, or if type is not compatible with
  > the type of the actual next argument (as promoted according to
  > the default argument promotions), the behavior is undefined, except for
  > the following cases:...

- gcc's behavior: random value from memory to stdout / null and 0 to stdout

### [2. Signed integer overflow](./02_integer-overflow/)

- Source: para. 3 of section 3.4.3 of [C11][1]:

  > An example of undefined behavior is the behavior on integer overflow.

- gcc's behavior:

  - `-O1`: `INT_MAX + 1 == INT_MIN`, i.e., `2147483647 + 1` becomes `-2147483648`
  - `-O2`/`-O3`: still observe the above, but the loop won't quit after `i`
    reaches `100`. The program will be trapped in an infinite loop.
  - This is likely because gcc relies on the no-overflow assumption
    to optimize how to quit a loop. [[2][2]]

- Why don't we define it?
  - Leaving signed integer overflow undefined opens the door for various
    optimizations, such as converting `x * 2 / 2` to `x` and `x + 1 > x` to
    `true`.

### [3. Shift pass bit width/oversized shift amount](./03_shift-overflow/)

- Source: para. 3 of section 6.5.7 of [C11][1]:

  > If the value of the right operand is negative or is greater than or equal
  > to the width of the promoted left operand, the behavior is undefined

- gcc's behavior: `1 << 35 == 8`, which is equal to
  `1 << (35 % (sizeof(int) * CHAR_BIT))` or `1 << (35 % 32)`.

- Why don't we define it?
  - [One source][2] says that this originated because the underlying
    shift operations on various CPUs do different things with this: for example,
    X86 truncates 32-bit shift amount to 5 bits (so a shift by 32-bits is
    the same as a shift by 0-bits), but PowerPC truncates 32-bit shift amounts
    to 6 bits (so a shift by 32 produces zero). Because of these hardware
    differences, the behavior is completely undefined by C (thus shifting by
    32-bits on PowerPC could format your hard drive, it is _not_ guaranteed
    to produce zero)

### 4. Pass a non-null-terminated C-string to `strlen()`

- Source: para. 3 of section 7.24.6.3 of [C11][1]:

  > The `strlen` function returns the number of characters that precede the
  > terminating null character.

  - This definition falls within the 2nd part of the definition of undefined
    behaviors:
    > by the omission of any explicit definition of behavior.

- One of gcc's possible behaviors: given the nature of this UB, no
  concrete test is performed. Theoretically, results could vary depending
  on where the pointer currently points to and whether a '\0' is near the bound
  of the c-string.

### [5. Use of uninitialized unsigned integer](./05_use-of-uninitialized-variable)

- As unsigned integer types don't have trap representation (i.e., all possible
  bit combinations are valid values), programers may think that using
  uninitialized unsigned integer will be spared from undefined behaviors. The
  implicit argument is that no matter what the uninitizlied memory blocks
  contain, the wrost case scenario is we read some rubbish bits (yet still
  valid unsigned int) from it.

  - Unfortunately, the trust is misplaced. This is because compilers are
    free to **not** reserving memory blocks for uninitilized variables, rendering
    merely reading from them (i.e., non-existent memory) undefined.

- Source: para. 2 of section 6.3.2.1 of [C11][1]:

  > If the lvalue designates an object of automatic storage duration that
  > could have been declared with the register storage class (never had its
  > address taken), and that object is uninitialized (not declared with an
  > initializer and no assignment to it has been performed prior to use),
  > the behavior is undefined.

- This is one of the more jargon-heavy definitions of undefined behaviors.

- lvalue: this is a loaded concept. Long story short, if an expression can
  appear on the left-hand side of `=`, it is an lvalue; otherwise it is an
  rvalue. For example, we define `int a = 0, b = 1`:
  - Variable `a` is an lvalue, as we can do `a = 3;`
  - Expression `a + b` is an rvalue, as it doesn't make sense to have
    `a + b = 3`;
  - `3` is also an rvalue.
- Automatic storage duration: if a variable is with automatic storage
  duration, it roughly means that the variable is stored on "stack".
  - But what does "stack" mean anyway? It has to do with the popular
    instruction pointer protocol supported by common `call` and `ret`
    instructions. It doesn't tell us anything about the lifetime of an
    object, except through a historical association to object lifetimes in
    C, due to popular stack frame conventions.
- Register storage class: is used to define local variables that should
  be stored in a register instead of RAM. This implies two important things:
  1. The variable has a maximum size equal to the register size (usually
     one word);
  1. The variable can't have the unary `&` operator applied to it. That is,
     for a variable `a`, `&a` is illegal as it doesn't have an address since
     the very beginning.
- never had its address taken: for variable `a`, we have never done `&a`.
  "could have been declared with the register storage class" and "never had its
  address taken" should mean the same.

- One of gcc's possible behaviors: variables that are not explicitly
  initialized are implictly initialized as `0`.

### [6. Float to unsigned integer conversion](./06_float-to-uint-conversion/)

- As the range of float and signed integer are different, it is not too
  surprising that float to signed integer conversion could invoke UB. However,
  as documented in the
  [Seemingly undefined by actually well-defined behaviors](#seemingly-undefined-by-actually-well-defined-behaviors)
  section below, one may be tempted to think that converting float/double to
  unsigned integer type is always defined.

  - This is not the case though.

- Source: para. 1 of section 6.3.1.4 of [C11][1]:

  > When a finite value of real floating type is converted to an integer
  > type other than **\_Bool**, the fractional part is discarded (i.e., the
  > value is truncated toward zero). If the value of the integral part
  > cannot be represented by the integer type, the behavior is undefined.
  > <sup>61)</sup>

  and footnote 61:

  > The remaindering operation performed when a value of integer type is
  > converted to unsigned type need not be performed when a value of real
  > floating type is converted to unsigned type. Thus, the range of
  > portable real floating values is (−1, U*type*\_MAX+1)

- While being a bit long-winded and unexpected, the 1st paragraph of section
  6.3.1.4 is relatively clear. It means the following:

  ```C
  float    a = 3.14;
  uint32_t b = (uint32_t)a;
  // defined, b == 3

  float    a = -1.23;
  uint32_t b = (uint32_t)a;
  // UB, as the integral part, -1, can't be represented by uint32_t

  float a = 2147483646.0;
  // defined
  uint32_t b = (uint32_t)a;
  // defined, b == 2147483646
  uint8_t  c = (uint8_t )a;
  // UB, as the integral part, 2147483646, can't be represented by uint8_t
  ```

- The footnote's "the range of portable real floating values is
  (−1, U*type*\_MAX+1)" is a bit more confusing. It means the following:

  - Parenthesis (i.e., `(` and `)`) in (−1, U*type*\_MAX+1) just means that both
    end of the range is exclusive. So for `uint8_t`, whose range is [0, 255],
    a "portable" float's range is (-1, 255+1).
  - But why this is the case? `-0.9` and `255.5` are beyond the range of
    `uint8_t` already, how come it doesn't cause UB?
  - One has to read the standard twice to get the gist--section 6.3.1.4
    says that

    > the fractional part is discarded (i.e., the value is truncated toward
    > zero).

    as `-0.9` becomes `0` and `255.5` becomes `255`, they are still within
    the range and representable by `uint8_t`.

- But why does C standard has two seemingly inconsistent way to handle
  `unsigned int`?

  - The reason for this apparent semantic inconsistency in the C Standard is
    probably the concern to keep existing implementations and hardware behavior
    compatible with the Standard. It may be linked to the binary representation
    of negative floating point numbers: while all but some ancient
    architectures have used two's complement representation for signed integers
    for a long time, floating point numbers generally use sign + magnitude
    representations. The modulo semantics of signed integer to unsigned integer
    conversions costs nothing on two's complement representations, but would
    require extra silicon for floating point values, which was not present on
    all current hardware implementations at the time. The Standard Committee
    decided to keep these cases undefined for `uint32_t = (uint32_t)-1.23;`
    and also for the less problematic `uint8_t a = (uint8_t)123456.7;` to
    avoid the requirement for compiler writers to produce extra costly code
    to fix the behavior on hardware that does not implement the modulo
    semantics already.[\[6\]][6]

- gcc's behavior (is neither stable nor predictable):

  ```shell
  ./main.out
  Converting -3.140000 to uint32_t gives 4294967293
  Converting 2147483648.000000 to uint8_t gives 0

  ./main-o1.out
  Converting -3.140000 to uint32_t gives 0
  Converting 2147483648.000000 to uint8_t gives 255

  ./main-o2.out
  Converting -3.140000 to uint32_t gives 0
  Converting 2147483648.000000 to uint8_t gives 255

  ./main-o3.out
  Converting -3.140000 to uint32_t gives 0
  Converting 2147483648.000000 to uint8_t gives 255
  ```

### [7. Strict aliasing rule violation (e.g. Dereferencing a float pointer as an unsigned int pointer)](./07_strict-aliasing-rule-violation)

- This is yet another nuanced case. The general idea is that doing the
  following causes UB:

  ```C
  // Let's assume sizeof(float) == sizeof(unsigned int) == 4
  float pi = 3.14;
  unsigned int* pi_int = (unsigned int*)&pi;
  ```

  even if we guarantee that `unsigned int` does not have any trap
  representation.

  - Well this guarantee is not true either. C standard only guarantees
    that `unsigned char` does not have any trap representation (in section
    6.2.6.2 §2 of [C11][1]).

- Source: para. 1 of section 6.3.1.4 of [C11][1]:

  > An object shall have its stored value accessed only by an lvalue
  > expression that has one of the following types: [88]
  >
  > — a type compatible with the effective type of the object,
  >
  > — a qualified version of a type compatible with the effective type of
  > the object,
  >
  > — a type that is the signed or unsigned type corresponding to the
  > effective type of the object,
  >
  > — a type that is the signed or unsigned type corresponding to a
  > qualified version of the effective type of the object,
  >
  > — an aggregate or union type that includes one of the aforementioned
  > types among its members (including, recursively, a member of a
  > subaggregate or contained union), or
  >
  > — a character type.
  >
  > ***
  >
  > [88] The intent of this list is to specify those circumstances in which
  > an object may or may not be aliased

  - **Compatible** is explained in section 6.2.7 of [C11][1].
    Two types are "compatible" only if they are "the same". For example,
    `int32_t` and `int` are compatible and `int` and `float` are not compatible.
    `int` and `short` are not compatible either.
  - **Qualified** is explained in section 6.7.3 of [C11][1]. Essentially,
    a qualified version of a type means a type with qualifiers. Qualifiers
    are keywords like `const`, `volatile`, etc.
  - A more detailed explanation can be found
    [here](https://www.cs.auckland.ac.nz/references/unix/digital/AQTLTBTE/DOCU_020.HTM)
    .

  - The standard lists a few cases where aliasing are allowed. Going through
    them one by one could be a bit tedious. To summerize, only the following
    and their obvious variants are legal aliasing:

    ```C

    int x = 1;
    int *p = &x;
    signed int *p = &x;                      // Compatible type
    const int *p = &x;                       // Qualified version
    unsigned char *p = (unsigned char *)&x;  // Character type
    ```

    while the following and most other unmentioned ones are illegal:

    ```C
    int x = 1;
    short *p = (short*)&x;
    float *p = (float*)&x;
    double *p = (double*)&x;
    ```

- The rationale behind the strict aliasing rule is performance. By assuming
  that different type can't be aliased, compilers can apply a wide range of
  optimization techniques called "Type-Based Alias Analysis" (TBAA).

- Two examples are prepared to demonstrate the effect of TBAA
  [here](./07_strict-aliasing-rule-violation/lib.c):

  ```C
  void manipulate_inplace_int(int* arr, int* y, size_t arr_size) {
      for (int i = 0; i < arr_size; ++i)
          arr[i] = *y + 42;
  }

  void manipulate_inplace_float(int* arr, int16_t* y, size_t arr_size) {
      for (int i = 0; i < arr_size; ++i)
          arr[i] = *y + 42;
  }
  ```

- These two functions differ by the type of `y` only. This trival difference
  results in different [machine code](./07_strict-aliasing-rule-violation/lib.asm):

  ```asm
  void manipulate_inplace_int(int* arr, int* y, size_t arr_size) {
      for (int i = 0; i < arr_size; ++i)
    0:	test   rdx,rdx
    3:	je     21 <manipulate_inplace_int+0x21>
    5:	lea    rdx,[rdi+rdx*4]
    9:	nop    DWORD PTR [rax+0x0]
          arr[i] = *y + 42;
    10:	mov    eax,DWORD PTR [rsi]
      for (int i = 0; i < arr_size; ++i)
    12:	add    rdi,0x4
          arr[i] = *y + 42;
    16:	add    eax,0x2a
    19:	mov    DWORD PTR [rdi-0x4],eax
      for (int i = 0; i < arr_size; ++i)
    1c:	cmp    rdi,rdx
    1f:	jne    10 <manipulate_inplace_int+0x10>
  }
    21:	ret
    22:	data16 nop WORD PTR cs:[rax+rax*1+0x0]
    2d:	nop    DWORD PTR [rax]
  ```

  ```asm
  void manipulate_inplace_short(int* arr, int16_t* y, size_t arr_size) {
      for (int i = 0; i < arr_size; ++i)
    30:	test   rdx,rdx
    33:	je     4b <manipulate_inplace_short+0x1b>
          arr[i] = *y + 42;
    35:	movsx  eax,WORD PTR [rsi]
    38:	lea    rdx,[rdi+rdx*4]
    3c:	add    eax,0x2a
    3f:	nop
    40:	mov    DWORD PTR [rdi],eax
      for (int i = 0; i < arr_size; ++i)
    42:	add    rdi,0x4
    46:	cmp    rdi,rdx
    49:	jne    40 <manipulate_inplace_short+0x10>
  }
    4b:	ret
  ```

  - The difference is about where `42` (`0x2a`) is added to `eax`.

  1. In the `int`'s version, it is added at `16`, inside the loop, meaning
     that the calculation is performed in each iteration. This is because
     `arr` and `y` is of the same type, compilers can't rule out the possibility
     that `arr` and `y` refer to a common memory location.
  1. In the `short`'s version, it is added at `3c`, meaning that the calculation
     is performed only once, before the loop. This is because per C standard
     `arr` and `y` are not of compatible types, so that they can't be referring
     to the same memory location.

- This rule has another very significant implication. There has long been a
  pain point in C that we don't have a proper `byte` type. Typically people
  might use `uint8_t` as a more self-explanatory alternative of `byte` or use
  the more common `unsigned char` type.
  - Using `uint8_t` is, strictly speaking, a violation of the strict aliasing
    rule. As 6.3.1.4 of [C11][1] provides, the standard-complain way to access
    a variable must be done via either a "compatible" type or a character type.
    Usint `uint8_t` to access memory with known data type could lead to undefined
    behaviors.
  - Paragraph 15 of section 6.2.5 of [C11][1] stipualtes that types `char`,
    `signed char`, and `unsigned char` are collectively called the
    _character types_.

### 8. Arbitrary pointers comparison

- Suppose you have a range of memory described by two variables, say,

  ```C
  byte* regionStart;
  size_t regionSize;
  ```

  and suppose you want to check whether a pointers lies within that region.
  You might be tempted to write:

  ```C
  if (p >= regionStart && p < regionStart + regionSize)
  ```

- This comparison is undefined.

- 6.5.8 §5 of [C11][1] stipulates that:

  > When two pointers are compared, the result depends on the relative
  > locations in the address space of the objects pointed to. If two
  > ointers to object types both point to the same object, or both point
  > one past the last element of the same array object, they compare equal.
  > If the objects pointed to are members of the same aggregate object,
  > pointers to structure members declared later compare greater than
  > pointers to members declared earlier in the structure, and pointers to
  > array elements with larger subscript values compare greater than
  > pointers to elements of the same array with lower subscript values.
  > All pointers to members of the same union object compare equal. If the
  > expression P points to an element of an array object and the expression
  > Q points to the last element of the same array object, the pointer
  > expression Q+1 compares greater than P. In all other cases, the
  > behavior is undefined.

- This paragraph lays out a few scenarios where pointer-to-pointer comparisons
  are defined, and left all other cases undefined.

  - Unfortunately, the use of
    `(p >= regionStart && p < regionStart + regionSize)` falls outside of the
    defined scenarios in the standard.

- Why this is the case? Raymond Chen gives a very good explaination [here][7].
  - Long story short, it is because the memory layout is not always flat,
    so for a given architecture, `*(p++)` does not necessarily correspond to
    moving to next memory address. As a result, C standard intentionally leaves
    this as undefined.

### 9. Constructs using pre and post-increment

- Many (if not all) of these funny statements cause undefined behaviors:

  ```C
  i = (i++);
  i = i++ + ++i;
  i = i++ + i ++;
  i += i *= i;
  ```

- Section 6.5 §2 of [C11][1] stipulates that:
  > If a side effect on a scalar object is unsequenced relative to either
  > a different side effect on the same scalar object or a value computation
  > using the value of the same scalar object, the behavior is undefined.
  - Per this [SO post](https://stackoverflow.com/a/31083924/19634193),:
    - "Value computations" means to work out the result of an expression;
    - "Side effects" means the modifications of objects.
- 6.5.16 §3 of [C11][1] stipulates that:
  > An assignment operator ...The evaluations of the operands are unsequenced.
- TL;DR: trying to modify `i` twice in a statement without a sequence point
  causes UB.

## Seemingly undefined by actually well-defined behaviors

- `unsigned int` never overflow, for `unsigned int a = UINT_MAX;`, `a + 1`
  will be "wrapped around", i.e., to `(a + 1) % UINT_MAX == 0`.
  - However, `unsigned int a = 2147483646 * 2147483646;` may lead to undefined
    behaviors since `2147483646 * 2147483646` could be considered as signed
    multiplication before the signed result is cast and assigned to
    `unsigned int a`. To make sure it is unsigned, rewrite it as
    `2147483646U * 2147483646U`.

## References

1. [Draft of ISO C11 Standard][1]
1. [What Every C Programmer Should Know About Undefined Behavior #1/3][2]
1. [(Why) is using an uninitialized variable undefined behavior?][3]
1. [Why are the terms "automatic" and "dynamic" preferred over the terms "stack" and "heap" in C++ memory management?][4]
1. ["A Guide to Undefined Behavior in C and C++, Part 1][5]
1. [How to intrepret paragraph 1 of section 6.3.1.4 of C11 standard (about converting float to unsigned int)][6]
1. [How to check if a pointer is in a range of memory][7]

[1]: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf "Draft of ISO C11 Standard"
[2]: https://blog.llvm.org/2011/05/what-every-c-programmer-should-know.html "What Every C Programmer Should Know About Undefined Behavior #1/3"
[3]: https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior "(Why) is using an uninitialized variable undefined behavior?"

[4]: https://stackoverflow.com/questions/9181782/why-are-the-terms-automatic-and-dynamic-preferred-over-the-terms-stack-and "Why are the terms \"automatic\" and \"dynamic\" preferred over the terms \"stack\" and \"heap\" in C++ memory management?"
[5]: https://blog.regehr.org/archives/213 "A Guide to Undefined Behavior in C and C++, Part 1"
[6]: https://stackoverflow.com/questions/75578931/how-to-intrepret-paragraph-1-of-section-6-3-1-4-of-c11-standard-about-convertin/ "How to intrepret paragraph 1 of section 6.3.1.4 of C11 standard (about converting float to unsigned int)"
[7]: https://devblogs.microsoft.com/oldnewthing/20170927-00/?p=97095 "How to check if a pointer is in a range of memory"
