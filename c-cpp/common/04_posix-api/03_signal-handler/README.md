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

* 

## References

1. [Introduction To Unix Signals Programming][1]
1. [Wikipedia - C signal handling][2]

[1]: http://www.cs.kent.edu/~ruttan/sysprog/lectures/signals.html "Introduction To Unix Signals Programming"
[2]: https://en.wikipedia.org/wiki/C_signal_handling "C signal handling"