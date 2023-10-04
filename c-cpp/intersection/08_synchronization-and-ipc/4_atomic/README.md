# Atomic

## C

- According to 8.1.1 of the
  [Intel® 64 and IA-32 Architectures Software Developer’s Manual](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3a-part-1-manual.pdf)
  and the 24.4.7.2 of the
  [The GNU C Library (glibc) manual](https://www.gnu.org/software/libc/manual/html_node/Atomic-Types.html)
  `int` without any qualifiers is mostly atomic already.

- `sig_atomic_t`, an integer type which can be accessed as an atomic entity
  even in the presence of asynchronous interrupts made by signals, is just
  an alias of `int` defined in `sig_atomic_t.h` and `types.h`:

  ```C
  typedef int __sig_atomic_t;
  typedef __sig_atomic_t sig_atomic_t;
  ```

- https://lumian2015.github.io/lockFreeProgramming/c11-features-in-currency.html
