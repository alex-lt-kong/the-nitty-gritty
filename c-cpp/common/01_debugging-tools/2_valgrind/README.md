# Valgrind

* `apt install valgrind`

## Limitations

* it is quite slow;
* it can only find bugs that still exist in the generated machine code (so it
can't find things the optimizer removes);
* it and doesn't know that the source language is C (so it can't find
shift-out-of-range or signed integer overflow bugs).

## References

* [What Every C Programmer Should Know About Undefined Behavior #2/3](https://blog.llvm.org/2011/05/what-every-c-programmer-should-know_14.html)