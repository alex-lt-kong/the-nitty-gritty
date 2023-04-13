# Multithreading and Synchronization

* Native multithreading support is one of the more exciting new features
of C++11.
    * Prior to that, C++ developers have to use platform-specific models,
    such as POSIX's threads (pthreads) to implement multithreading,
    just like my projects [here](../../common/04_posix-api/03_signal-handler/)
    and [here](https://github.com/alex-lt-kong/camera-server).
    * This works fine and can help developers understand low-level details.
    But it makes portability more difficult and imposes greater challenges on
    the weak-minded.