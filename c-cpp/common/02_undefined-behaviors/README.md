# Undefined behaviors

* While the general idea of undefined behavior is not difficult to understand,
the exact wording may vary. [C11 standard][1] defines undefined behavior as
follows:
  > undefined behavior is otherwise indicated in this International Standard by
  > the words ‘‘undefined behavior’’ or by the omission of any explicit
  > definition of behavior.
* In the C community, undefined behavior may be humorously referred to as
"nasal demons", after a comp.std.c post that explained undefined behavior as
allowing the compiler to do anything it chooses, even "to make demons
fly out of your nose".
* It is common for programmers, even experienced ones, to rely on undefined
behavior either by mistake, or simply because they are not well-versed in
the rules of the language that can span hundreds of pages. This can result
in bugs that are exposed when a different compiler or lead to security
vulnerabilities in software.

* But if undefined behaviors are so bad, why don't we just define them?
  * Documenting an operation as undefined behavior allows compilers to assume
  that this operation will never happen in a conforming program. This gives
  the compiler more information about the code and this information can lead to
  more optimization opportunities. 
  * Simply put, the existence of undefined behaviors makes C fast!

## Some common undefined behaviors


### [1. Format specifier without argument](./01_format-specifier-without-argument/)

* Source: para. 2 of section of 7.16.1.1 of [C11][1]:

  > If there is no actual next argument, or if type is not compatible with
  > the type of the actual next argument (as promoted according to
  > the default argument promotions), the behavior is undefined, except for
  > the following  cases:...

* gcc's behavior: random value from memory to stdout / null and 0 to stdout


### [2. Signed integer overflow](./02_integer-overflow/)

* Source: para. 3 of section 3.4.3 of [C11][1]:
  > An example of undefined behavior is the behavior on integer overflow.

* gcc's behavior:
  * `-O1`: `INT_MAX + 1 == INT_MIN`, i.e., `2147483647 + 1` becomes `-2147483648`
  * `-O2`/`-O3`: still observe the above, but the loop won't quit after `i`
  reaches `100`. The program will be trapped in an infinite loop.
  * This is likely because gcc relies on the no-overflow assumption
  to optimize how to quit a loop. [[2][2]]

* Why don't we define it?
  * Leaving signed integer overflow undefined opens the door for various
  optimizations, such as converting `x * 2 / 2` to `x` and `x + 1 > x` to
  `true`.


### [3. Shift pass bit width/oversized shift amount](./03_shift-overflow/)

* Source: para. 3 of section 6.5.7 of [C11][1]:
  > If the value of the right operand is negative or is greater than or equal
  > to the width of the promoted left operand, the behavior is undefined

* gcc's behavior: `1 << 35 == 8`, which is equal to
`1 << (35 % (sizeof(int) * CHAR_BIT))` or `1 << (35 % 32)`.

* Why don't we define it?
  *  [One source][2] says that this originated because the underlying
  shift operations on various CPUs do different things with this: for example,
  X86 truncates 32-bit shift amount to 5 bits (so a shift by 32-bits is
  the same as a shift by 0-bits), but PowerPC truncates 32-bit shift amounts
  to 6 bits (so a shift by 32 produces zero). Because of these hardware
  differences, the behavior is completely undefined by C (thus shifting by
  32-bits on PowerPC could format your hard drive, it is *not* guaranteed
  to produce zero)


### 4. Pass a non-null-terminated C-string to `strlen()`

* Source: para. 3 of section 7.24.6.3 of [C11][1]:
  > The `strlen` function returns the number of characters that precede the
  > terminating null character.

  * This definition falls within the 2nd part of the definition of undefined
  behaviors:
    > by the omission of any explicit definition of behavior.

* gcc's behavior: given the nature of this UB, no concrete test is performed.
Theoretically, results could vary depending on where the pointer currently
points to and whether a '\0' is near the bound of the c-string.


### [5. Use of unintilized unsigned integer](./05_use-of-uninitialized-variable)

* Source: para. 2 of section 6.3.2.1 of [C11][1]:
  > If the lvalue designates an object of automatic storage duration that
  > could have been declared with the register storage class (never had its
  > address taken), and that object is uninitialized (not declared with an
  > initializer and no assignment to it has been performed prior to use),
  > the behavior is undefined.

* This is one of the more jargon-heavy definitions of undefined behaviors.

* lvalue: this is a loaded concept. Long story short, if an expression can
appear on the left-hand side of `=`, it is an lvalue; otherwise it is an
rvalue. For example, we define `int a = 0, b = 1`:
  * Variable `a` is an lvalue, as we can do `a = 3;`
  * Expression `a + b` is an rvalue, as it doesn't make sense to have
  `a + b = 3`;
  * `3` is also an rvalue.
* Automatic storage duration: if a variable is with automatic storage
duration, it roughly means that the variable is stored on "stack".
  * But what does "stack" mean anyway? It has to do with the popular
  instruction pointer protocol supported by common `call` and `ret`
  instructions. It doesn't tell us anything about the lifetime of an
  object, except through a historical association to object lifetimes in
  C, due to popular stack frame conventions.
* Register storage class: is used to define local variables that should
be stored in a register instead of RAM. This implies two important things:
  1. The variable has a maximum size equal to the register size (usually
  one word);
  1. The variable can't have the unary `&` operator applied to it. That is,
  for a variable `a`, `&a` is illegal as it doesn't have an address since
  the very beginning.
* never had its address taken: for variable `a`, we have never done `&a`.
"could have been declared with the register storage class" and "never had its
address taken" should mean the same.


* gcc's behavior: variables that are not explicitly initialized are implictly
initialized as `0`.

## Seemingly undefined by actually well-defined behaviors

* `unsigned int` never overflow, for `unsigned int a = UINT_MAX;`, `a + 1`
will be "wrapped around", i.e., `(a + 1) % UINT_MAX == 0`.
  * However, `unsigned int a = 2147483646 * 2147483646;` may lead to undefined
  behaviors since `2147483646 * 2147483646` could be considered as signed
  multiplication before the signed result is cast and assigned to
  `unsigned int a`. To make sure it is unsigned, rewrite it as
  `2147483646U * 2147483646U`.

## References

1. [Draft of ISO C11 Standard][1]
1. [What Every C Programmer Should Know About Undefined Behavior #1/3][2]
1. [(Why) is using an uninitialized variable undefined behavior?][3]
1. ["Why are the terms "automatic" and "dynamic" preferred over the terms "stack" and "heap" in C++ memory management?"][4]

[1]: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf "Draft of ISO C11 Standard"
[2]: https://blog.llvm.org/2011/05/what-every-c-programmer-should-know.html "What Every C Programmer Should Know About Undefined Behavior #1/3"
[3]: https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior "(Why) is using an uninitialized variable undefined behavior?"
[4]: https://stackoverflow.com/questions/9181782/why-are-the-terms-automatic-and-dynamic-preferred-over-the-terms-stack-and "Why are the terms \"automatic\" and \"dynamic\" preferred over the terms \"stack\" and \"heap\" in C++ memory management?"
