# Polymorphism

- The first point to note is that "polymorphism" comes
  in [many forms](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#Forms).
  Two popular forms are:
    - Function overloading, a.k.a., ad hoc polymorphism
    - Function overriding, a.k.a., subtyping

## Function overloading

- With function overloading, multiple functions can have the same name with
  different
  parameters.[[1](https://www.w3schools.com/cpp/cpp_function_overloading.asp)]

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

- Function overriding requires the use of virtual function: a virtual function
  is a member function that is declared within a base class and is re-defined (

- Unlike function overloading, function overriding indeed requires virtual
  table (vtable). Note that vtable is not something specified in the C++
  standard but is a common implementation of polymorphism in
  C++.[[2](https://dev.to/pgradot/vtables-under-the-surface-3foa)]
  overridden) by a derived class.

- C++'s does not have a stable Application Binary Interface (ABI), but "most
  major compilers (except MSVC) follow the Itanium C++
  ABI"[[3](https://dev.to/pgradot/vtables-under-the-surface-3foa)]

    - Itanium C++ The ABI has a section about vtables, so using any compiler
      following this ABI should yield similar implementation details.

    - But we are not going to delve into the details of the ABI (as well as the
      vtable's layout) here.
    -
- We can use `objdump` to reveal the existence of vtable in the binary
  file.

```
>>> objdump --syms --demangle function-overriding | grep vtable
0000000000018d88  w    O .data.rel.ro   0000000000000028              vtable for std::format_error
0000000000018c80  w    O .data.rel.ro   0000000000000028              vtable for std::__format::_Iter_sink<char, std::__format::_Sink_iter<char> >
0000000000000000       O *UND*  0000000000000000              vtable for __cxxabiv1::__class_type_info@CXXABI_1.3
0000000000018c30  w    O .data.rel.ro   0000000000000028              vtable for Base
0000000000018ca8  w    O .data.rel.ro   0000000000000020              vtable for std::__format::_Formatting_scanner<std::__format::_Sink_iter<char>, char>
0000000000000000       O *UND*  0000000000000000              vtable for __cxxabiv1::__si_class_type_info@CXXABI_1.3
0000000000018c58  w    O .data.rel.ro   0000000000000028              vtable for Derived
0000000000018c08  w    O .data.rel.ro   0000000000000028              vtable for std::__format::_Seq_sink<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >
```

## Curiously Recurring Template Pattern (CRTP)

- But in C++ there is another way to avoid the overhead of vtable called CRTP (
  Curiously Recurring Template Pattern)