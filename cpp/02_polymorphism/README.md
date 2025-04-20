# Virtual Table

- The first point to note is that "polymorphism" comes
  in [many forms](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#Forms).
  Two popular forms are:
    - Function overloading, a.k.a., ad hoc polymorphism
    - Function overriding, a.k.a., subtyping

- Only function overriding requires virtual table (vtable), function overloading
  does not require vtable.
    - Refer
      to [this post](https://dev.to/pgradot/vtables-under-the-surface-3foa)

- But in C++ there is another way to avoid the overhead of vtable called CRTP (
  Curiously Recurring Template Pattern)