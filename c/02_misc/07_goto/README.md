# Goto

- While `goto` seems to be an epitome of C's flexibility and allows the program
  to jump anywhere, there are some limitations and implications on what
  it can and can't do.

- Bypassing variable initialization is fine--while the variable will be
  uninitialized, the program can run.

- Bypassing variable declaration is, surprisingly, fine as well, as long as
  the variable is not a variable-length array (VLA).

  - Originally, in C89, it was mandatory to declare all variables at the
    start of a block. This directly conveys a very important thing: one
    block has one unchanging set of variable declarations.
  - C99 changed this. Per C99, you can declare variables in any part of the
    block, but declaration statements are still different from regular statements.
    To understand this, one can imagine that all variable declarations are
    implicitly moved to the start of the block where they are declared and
    made unavailable for all statements that preceded them.
    - This is a bit similar but not exactly the same as JavaScript's
      [Hoisting](https://developer.mozilla.org/en-US/docs/Glossary/Hoisting)
  - As a result, even if your `goto` skips over a variable declaration, the
    program still works.

- For VLA (a.k.a. variably modified type), bypassing declaration won't work.

  - Since the below code is valid:
    ```C
    int a;
    a = 10;
    int arr[a];
    ```
  - What would you expect if it is skipped?:
    ```C
        goto output;
        int a;
        a = 10;
        int arr[a];
    output:
        printf("%d (%p)\n", arr[0], &arr);
    ```
  - So let's just ban it instead!

- There are some edge cases where code can be compiled in C but not in C++.
  But let's not touch them in this repo...

## Reference

- [Skip variable declaration using goto?](https://stackoverflow.com/questions/29880836/skip-variable-declaration-using-goto)
- [Is goto from outside a block past an initialized variable declaration guaranteed to give a compile error?](https://stackoverflow.com/questions/34081317/is-goto-from-outside-a-block-past-an-initialized-variable-declaration-guaranteed)
