# File descriptor, open() and fopen()

* `open()` is a system call while `fopen()` is a library function.

* `open()` returns an integer that identifies the file (a.k.a. a file descriptor); `fopen()` returns `FILE` pointer. 

* `open()` is a POSIX standard which may not be available on other platforms; `fopen()` is defined in `stdio.h` and
it is portable.

* With a `FILE` pointer, you can use functions like `fscanf()`, `fprintf()`, etc. If you have just the file descriptor,
you have limited (but likely faster) input and output routines, such as `read()`, `write()`, etc.

* Note that we can also `read()` data from and `write()` data to a socket.

* Many higher level functions can be replicated by directly `read()`ing from or `write()`ing pseudo device files
such as `/dev/stdin`, `/dev/stderr`, `/dev/tty` and `/dev/urandom`.