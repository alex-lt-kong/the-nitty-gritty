# sizeof(bool)

- It is interesting that people may be tempted to think we can use `bool` in C/C++ to save some memory.
  This is most likely a misunderstanding...

- § 6.2.5 of C standard stipulates that

  > 2\. An object declared as type \_Bool is large enough to store the values 0 and 1

  > 6\. The type \_Bool and the unsigned integer types that correspond to the standard
  > signed integer types are the standard unsigned integer types.

  - So strictly speaking C standard doesn't say how exactly a bool variable must be implemented and
    people are free to implement it using one bit only--if `bit` is addressable... In reality, it has to
    occupy one byte of space instead of one bit so it is no more space-efficient than `unsigned char`.

## References

- [N1256 Draft of C standard](https://www.open-std.org/jtc1/sc22/WG14/www/docs/n1256.pdf)
