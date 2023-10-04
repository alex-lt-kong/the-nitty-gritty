# Valgrind

* `apt install valgrind`

## Usage

* Refer to [Makefile](./Makefile)

* A useful [link](https://stackoverflow.com/questions/5134891/how-do-i-use-valgrind-to-find-memory-leaks)

## Limitations

* it is quite slow;
* it can only find bugs that still exist in the generated machine code (so it
can't find things the optimizer removes);
* it and doesn't know that the source language is C (so it can't find
shift-out-of-range or signed integer overflow bugs).

## References

* [What Every C Programmer Should Know About Undefined Behavior #2/3](https://blog.llvm.org/2011/05/what-every-c-programmer-should-know_14.html)