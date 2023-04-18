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

# [1_hello-world.cpp](./1_hello-world.cpp)

* This is the most naive version, not differs from pthreads too much.

# [2_join-and-detach.cpp](./2_join-and-detach.cpp)

* `join()`/`detach()` can be easily forgotten and it could cause confusing
errors.
    * One common error is "terminate called without an active exception" with
    a `SIGABRT` caught.

* The overall principle is very simple--a thread must be either `join()ed` or
`detach()ed` exactly once.
    * Why this is the case? Well the short answer is that C++ standard says
    so...[3]

* The existence of exceptions makes this a bit more complicated. Usually, we
do something like:

    ```C++
    th = thread(call_from_thread, 0);
    th.join();
    ```

    * However, we may, from time to time, add more statements between them:

    ```C++
    th = thread(call_from_thread, 0);
    spdlog::info("Thread started!");
    // more fancy operations.
    th.join();
    ```

    * If an exception is thrown between `thread()` and `join()`, we may
    have `try/catch` block that handles the exception，but `th` may be left
    un`join()ed`.

# [3_with-this.cpp](./3_with-this.cpp)

* The existence of the `this` pointer make multithreading more challenging--
this implementation showcases how we can access `this` in a thread.

* For the sake of completeness, two helper functions,
`static void signal_handler(int signum)` and `void install_signal_handler()`
are added to the PoC.

# [4_with-mutex.cpp](./4_with-mutex.cpp)

* The output of [3_with-this.cpp](./3_with-this.cpp) might be something like
this:
    ```
    [0] iterating...[1] iterating...
    [3] iterating...

    [5] iterating...
    [4[6] iterating...
    [7] iterating...
    [8] iterating...
    [9] iterating...
    ] iterating...
    [2] iterating...
    [5] iterating...
    [3] iterating...
    [8] iterating...
    [0] iterating...
    [9] iterating...
    [4] iterating...
    [[1] iterating...
    2[] iterating...
    [7] iterating...
    6] iterating...
    [5] iterating...
    [3] iterating...
    [8] iterating...
    [0] iterating...
    [1] iterating...
    [4] iterating...
    [2] iterating...
    [7] iterating...
    [9] iterating...[
    6] iterating...
    ```

* One could easily notice that the output is "corrupted" because different
threads try to write data to stdout.

* We may want to add `mutex` and `unique_lock` to prevent this:

    ```
    [0] iterating...
    [1] iterating...
    [5] iterating...
    [2] iterating...
    [3] iterating...
    [4] iterating...
    [6] iterating...
    [7] iterating...
    [8] iterating...
    [9] iterating...
    [1] iterating...
    [0] iterating...
    [4] iterating...
    [2] iterating...
    [7] iterating...
    [5] iterating...
    [6] iterating...
    [3] iterating...
    [8] iterating...
    [9] iterating...
    [1] iterating...
    [0] iterating...
    [2] iterating...
    [4] iterating...
    [7] iterating...
    [5] iterating...
    ```

* C++ provides three similar mutex ownership wrappers, `std::unique_lock`,
`std::lock_guard` and `std::scoped_lock`. 
    * `std::lock_guard` and `std::unique_lock` are pretty much the same thing;
    `std::lock_guard` is a restricted version with a limited interface.
    * A `std::lock_guard` always holds a lock from its construction to 
    its destruction. A `unique_lock` can be created without immediately locking,
    can unlock at any point in its existence, and can transfer ownership of the
    lock from one instance to another.[[1]]
    * `std::scope_lock` is a strictly superior version of `std::lock_guard`
    that locks an arbitrary number of mutexes all at once (using the same
    deadlock-avoidance algorithm as `std::lock`). In new code, you should
    only use `std::scoped_lock`.[[2]]


# [5_condition-var.cpp](./5_condition-var.cpp)

* We may have the below design from time to time:
    1. A few writing threads write to a shared object, say enqueuing new
    elements to a queue.
    1. One reading thread reads from a shared object, say, dequeuing one
    element from the queue at a time.
    1. If enqueue operations are frequent enough, we can just make the reading
    thread periodically.
    1. But if enqueue operations are sparse (or worse, unpredictable), it
    will be a waste of CPU resources if we poll the state of the queue
    too frequently and will cause great delay if we pool the state of the queue
    not frequently enough.

* Let's say there is one writing thread and one reading thread, the
implementation of the above awkward design will be like:

    * The writing thread:

    ```C++
    std::condition_variable cv;
    std::mutex stdout_mutex;
    std::queue<std::string> my_queue;

    while (true) {
        std::unique_lock<std::mutex> lk(my_mutex);
        // ... critical section ...    
        my_queue.push(msg);
    }
    ```

    * The reading thread:

    ```C++
    while (true) {
        std::unique_lock<std::mutex> lk(my_mutex);
        if (my_queue.empty()) {
            // Awkward, we keep polling the queue...
            this_thread::sleep_for(chrono::milliseconds(1000));
            continue;
        }        
        // ... critical section ...
        my_queue.pop();
    }
    ```


* What can be done to solve this elegantly? A condition variable comes to
rescue!

* Let's say we have two threads, a writing thread and a reading thread.

    * In the writing thread, we `notify_one()`/`notify_all()` reading threads:

    ```C++
    std::condition_variable cv;
    std::mutex stdout_mutex;
    std::queue<std::string> my_queue;

    while (true) {
        std::unique_lock<std::mutex> lk(my_mutex);
        // ... critical section ...    
        my_queue.push(msg);
        lk.unlock();
        cv.notify_one();
    }
    ```

    * In the reading thread, we `wait()` until we get notified.

    ```C++
    while (true) {
        std::unique_lock<std::mutex> lk(my_mutex);
        // Much better, we will wait() until we are notified.
        cv.wait(lk);
        // ... critical section ...
        my_queue.pop();
    }
    ```

## References

[1]: https://stackoverflow.com/questions/20516773/stdunique-lockstdmutex-or-stdlock-guardstdmutex "std::unique_lock<std::mutex> or std::lock_guard<std::mutex>?"
[2]: https://stackoverflow.com/questions/43019598/stdlock-guard-or-stdscoped-lock "std::lock_guard or std::scoped_lock?"
[3]: https://en.cppreference.com/w/cpp/thread/thread/~thread "std::thread::~thread"