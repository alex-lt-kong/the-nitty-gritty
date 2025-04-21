# Polymorphism

- The first point to note is that "polymorphism" comes
  in [many forms](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#Forms).
  Two popular forms are:
    - Function overloading, a.k.a., ad hoc polymorphism
    - Function overriding, a.k.a., subtyping

## Function overloading

- With function overloading, multiple functions can have the same name with
  different
  parameters.[1](https://www.w3schools.com/cpp/cpp_function_overloading.asp)

- Function overloading does not require virtual table (vtable) and is resolved
  at compile time.

- Issuing `objdump -t function-overloading | grep "F .text" | grep my_print`
  gives you the below:

  ```
  00000000000144c0  w    F .text	0000000000000694              _Z8my_printii
  0000000000013e30  w    F .text	0000000000000684              _Z8my_printi
  00000000000136d0  w    F .text	0000000000000754              _Z8my_printIJiiPKcS1_dEEvDpT_
  0000000000014b60  w    F .text	0000000000000664              _Z8my_printIJEEvDpT_
  ```

    - And after turning on name demangling, we can see the function
      signatures:

  ```
  >>> objdump --demangle  -t function-overloading | grep "F .text" | grep my_print
  00000000000144c0  w    F .text	0000000000000694              my_print(int, int)
  0000000000013e30  w    F .text	0000000000000684              my_print(int)
  00000000000136d0  w    F .text	0000000000000754              void my_print<int, int, char const*, char const*, double>(int, int, char const*, char const*, double)
  0000000000014b60  w    F .text	0000000000000664              void my_print<>()
  ```

## Function overriding

- Only function overriding requires virtual table (vtable), function overloading
  does not require vtable.
    - Refer
      to [this post](https://dev.to/pgradot/vtables-under-the-surface-3foa)

- But in C++ there is another way to avoid the overhead of vtable called CRTP (
  Curiously Recurring Template Pattern)