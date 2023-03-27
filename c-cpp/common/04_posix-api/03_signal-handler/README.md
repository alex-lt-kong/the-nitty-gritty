# Signals and signal handlers

* Signals are one of the most common IPC we use to interact with a running
program, mostly by pressing Ctrl-C (sending a `SIGINT` signal).
    * Also by running the `kill` command: `kill <signal> <pid>`.

*  Signals interrupt whatever the process is doing when they are sent, and
force the program to handle them immediately.

* Each signal has an integer number that represents it (1, 2 and so on),
as well as a symbolic name that is usually defined in `signal.h`.
    * Use `kill -l` to see a list of supported signals on the current system.


## Signals in C

* The C standard defines only 6 signals, in `signal.h`.
    * Additional signals may be specified in `signal.h` by the implementation.
    For example, Unix and Unix-like operating systems (such as Linux) define
    more than 15 additional signals.

## Signal handlers in C

* As we mostly use signals to gracefully exit an event loop, this project will
mainly focus on how to implement such a notification mechanism correctly.

* The `volatile` keyword: it indicates that a value may change between
different accesses, even if it does not appear to be modified.
    * More explicitly, it means volatile objects are read from memory each
    time their value is needed, and written back to memory each time they
    are changed.
    * Why does this matter? Consider the following example:
    ```C
    int a = getMyAValue();
    printf("a = %d\n");
    printf("a = %d\n");
    ```
    The compiler may think: oh as `a` is assigned only once, these two
    `printf()`s must result in the same string and it is not needed to
    read `a` from memory to register again, between two `printf()`s.
    * The `volatile` keyword prevents this from happening, as we are using
    a variable in a signal handler, it is not guaranteed that the value of
    `a` will remain the same if an event loop thread doesn't modify it (as
    a signal handler will change it sometimes, beyond the event loop
    thread).

* The `sig_atomic_t` type: it is the integer type of an object that can be
accessed as an atomic entity even in the presence of asynchronous interrupts.
    * The confusing part is that in many platforms `sig_atomic_t` type is just
    an alias of `int`. But how can we be sure that plain `int` always meets the
    atomicity requirement by the standard?
    * The trick is that a platform is free to pick a type to be used as
    `sig_atomic_t`. If a platform knows its `int` is standard compliant, it
    can simply make `sig_atomic_t` an alias to `int`. If on a specific
    platform its `int` doesn't meet the requirements, that specific platform
    has to come up with something else.

* While C standard does make it clear that a `volatile sig_atomic_t` variable
should be good enough in the presence of "asynchronous interrupts", it does
not explicitly mention if `volatile sig_atomic_t` is thread-safe.
    * One major reason is that the C standard does not define a multi-threading
    model prior to C11 and `volatile sig_atomic_t` exists way before that.
    * Therefore, the thread-safety of `volatile sig_atomic_t` is sort of
    implementation defined. Fortunately, as documented by glibc[7]:
        > In practice, you can assume that int is atomic. You can also
        > assume that pointer types are atomic; that is very convenient.
        > Both of these assumptions are true on all of the machines that
        > the GNU C Library supports and on all POSIX systems we know of. 
    so using `volatile sig_atomic_t` across different threads should be fine
    as long as we are on a "common" Linux platform.

* There is a oft-neglected rule that functions we can use in a signal
handler is very limited. According to C99 Rationale 2003:

    > the C89 Committee concluded that about the only thing a strictly
    > conforming program can do in a signal handler is to assign a value
    > to a volatile static variable which can be written uninterruptedly
    > and promptly return.[8]

    * Fortunately, POSIX standard relaxes this a bit, by allowing dozens
    (still not hundreds/thousands though) of "async-signal-safe" functions[9]
    to be called in a signal handler.

    * The attribute of "async-signal-safety" is closely related to another
    cryptic concept--"reentrancy"[10]. Roughly speaking, a function is
    re-entrant only if:
        * It does not use static or global variables, as they may be changed
        by time the function resumes;
        * It must not modify its own code;
        * If does not call any function that does not comply with the
        two rules above


* As discussed above, we can generally assume that on Linux
`volatile sig_atomic_t` can be used in a multh-threading environment, but
when a signal comes in, which thread will be used to handle the signal?
    * POSIX standard answers this unambiguously in `man 7 pthreads`:
        > POSIX.1 distinguishes the notions of signals that are directed
        > to the process as a whole and signals that are directed to
        > individual threads.  According to POSIX.1, a process-directed
        > signal (sent using `kill(2)`, for example) should be handled by a
        > single, arbitrarily selected thread within the process.

## References

1. [Introduction To Unix Signals Programming][1]
1. [Wikipedia - C signal handling][2]
1. [Wikipedia - volatile (computer programming)][3]
1. [IBM - The volatile type qualifier][4]
1. [SIG31-C. Do not access shared objects in signal handlers][5]
1. [SIG30-C. Call only asynchronous-safe functions within signal handlers][6]
1. [GNU libc - Atomic Types][7]
1. [Rationale for International Standard—Programming Languages—C][8]
1. [signal-safety(7) — Linux manual page][9]
1. [Wikipedia - Reentrancy (computing)][10]


[1]: http://www.cs.kent.edu/~ruttan/sysprog/lectures/signals.html "Introduction To Unix Signals Programming"
[2]: https://en.wikipedia.org/wiki/C_signal_handling "C signal handling"
[3]: https://en.wikipedia.org/wiki/Volatile_(computer_programming) "volatile (computer programming)"
[4]: https://www.ibm.com/docs/sr/zos/2.4.0?topic=qualifiers-volatile-type-qualifier "The volatile type qualifier"
[5]: https://wiki.sei.cmu.edu/confluence/display/c/SIG31-C.+Do+not+access+shared+objects+in+signal+handlers "SIG31-C. Do not access shared objects in signal handlers"
[6]: https://wiki.sei.cmu.edu/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers "SIG30-C. Call only asynchronous-safe functions within signal handlers"
[7]: https://www.gnu.org/software/libc/manual/html_node/Atomic-Types.html "GNU libc - Atomic Types"
[8]: www.open-std.org/jtc1/sc22/wg14/www/C99RationaleV5.10.pdf "Rationale for International Standard—Programming Languages—C"
[9]: https://man7.org/linux/man-pages/man7/signal-safety.7.html "signal-safety(7) — Linux manual page"
[10]: https://en.wikipedia.org/wiki/Reentrancy_(computing) "Reentrancy (computing)"