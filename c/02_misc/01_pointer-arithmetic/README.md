# Increment a pointer

- In C, a pointer is a variable that stores the value of the memory address of
  another variable. Adding an integer (positive or negative) to a pointer is
  valid in C.

  - In this sense, it is very similar to an unsigned integer--as the memory
    address it stores is mostly an unsigned integer.

- There is a catch though, for example:

```C
uint32_t arr[] = {0xFFFFFFFF, 0xEEEEEEEE, 0x11111111};
printf("%p:%x\n", (void*)arr, *arr);
printf("%p:%x\n", (void*)(arr+1), *(arr+1));
```

- If the first `printf()` outputs `0x7ffda81e23a4:deadbeef`, what would be
  the output of the second `printf()`? Would it be `0x7ffda81e23a8:eeeeffff`
  or `0x7ffda81e23a5:eeeeffff`

- The answer is `0x7ffda81e23a8:eeeeffff`.

- This means that `ptr++` is different from `i++` which increments the value
  by 1.

  - For pointers, `ptr++` increments it by the `sizeof()` its contents, that is,
    you're incrementing it as if you were iterating in an array.

- Note that the sample code in `main.c` may invoke undefined behaviors
  by violating the strict-aliasing rules.

- In standard C, addition and subtraction are not allowed on `void *`, since the
  target type’s size is not defined in that case. Likewise, they are not allowed
  on pointers to function types. However, these operations work in GNU C, and the
  "size of the target type" is taken as 1 byte.

## References

- [GNU C Language Introduction and Reference Manual - 14.10 Pointer Arithmetic](https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Pointer-Arithmetic.html)
