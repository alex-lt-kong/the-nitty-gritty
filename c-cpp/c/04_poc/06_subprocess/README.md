# Subprocess that redirects stdout/stderr with pipe()/fork()/exec()/poll()

* The purpose of the PoC is to develop a function that:
  1. takes the path and arguments of another program (a.k.a. child process)
  as its arguments;
  1. runs the child process (`sub.out`);
  1. wait for the child process to run and in the meantime capture child
  process's stdout, stderr and return code;
  1. resume its execution after the child process quits;
  Also, it should:
  1. be thread-safe, meaning that we can run the function concurrently
  without causing issues;
  1. not leak any resources, regardless of the behavior of the child
  process;
  1. not crash even if child process crashed.

* It seems that it is difficult to find a reference implementation for this
particular case online...So let's create one ourselves!
  * This turns out to be not an easy task...


* The first version is `main-naive.c`, it works, mostly. However, its
sequential `while()` loop structures means that if the child process
sent data to both stderr and stderr fast, the buffer for stderr will be filled
very soon, causing unexpected result.

* A more proper version is `main-poll.c`, it sets a reasonably large buffer
for potential child process's stdout and stderr and uses `poll()` to
alternatively read data from them.

* To throughly examine the behavior of the parent program, a sample child
program, `sub.out` is also prepared. It supports a few different modes
to demonstrate the interaction between child and parent programs.


## Known issues

* `main-poll.out`: 
  ```C
  if (close(pfds[j].fd) == -1)
      perror("close()");
  ```
  almost always triggers `EBADF: Bad file descriptor`. However, commenting this
  out causes some file descriptors to be unclosed, leaking resources.

## Useful notes

* We can check the opened file descriptors via: `ls -alh "/proc/$(pgrep main-poll)/fd"`