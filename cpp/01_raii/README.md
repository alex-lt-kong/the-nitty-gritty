# Resource acquisition is initialization

* RAII is an awkward name for a useful feature. Essentially, it is equivalent
to (if not better than) a `finally` keyword if properly used.

* Let's consider this Python snippet:

    ```python
        try:
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            # db operations
        except Exception as ex:
            print(ex)
        finally:
            if conn is not None:
                conn.close()
    ```

    * `finally` is needed because we don't want `conn` to be left open whether
    or not an exception is thrown.


* C++'s answer is rather straightforward, let's wrap the resource releasing
in destructor and always call destructor even if an exception is thrown:

    ```C++
        class DatabaseHelper {
        public:
        DatabaseHelper(RawDBConn* rawConn_) : rawConn(rawConn_) {};
        ~DatabaseHelper() {delete rawConn; }
        ... // omitted operator*, etc
        private:
        RawDBConn* rawConn;
        };

        DatabaseHelper handle(createNewResource());
        handle->performInvalidOperation();
    ```

    * The "official" opinion from
    [Bjarne Stroustrup](https://www.stroustrup.com/bs_faq2.html#finally)
    also argues that RAII is almost always better than a "finally" keyword.

    * Theoretically, RAII means "to bind the life cycle of a
    resource that must be acquired before use (allocated heap memory, thread
    of execution, open socket, open file, locked mutex, disk space,
    database connection—anything that exists in limited supply) to the lifetime
    of an object. ".
        * Microsoft summarizes the RAII principle as "to give ownership of any
        heap-allocated resource—for example, dynamically-allocated memory or
        system object handles—to a stack-allocated object whose destructor
        contains the code to delete or free the resource and also any
        associated cleanup code."

    * To be specific, RAII encapsulates each resource into a class, where:
        * the constructor acquires the resource and establishes all class
        invariants or throws an exception if that cannot be done,
        * the destructor releases the resource and never throws exceptions; 


* What if constructors or destructors throw exceptions? Things will get a bit
messier here.

    * In case of constructors:
        * If a constructor throws an exception, the object’s destructor is *not*
        run. If your object has already done something that needs to be undone
        (such as allocating some memory, opening a file, or locking a semaphore),
        this "stuff that needs to be undone" must be remembered by a data member
        inside the object.
    * In case of destructors:
        * Similar to Python's Exception raised in `finally` section, if exception
        is thrown within destructors, it has to be handled manually. This is beyond
        the scope of RAII.

* Apart from being handy for database connection management, etc
RAII is also the underlying principle of [smart pointers](../10_smart-pointers).
Say we use a class to wrap a raw pointer, using constructor to `malloc()`
memory and use its destructor to `free()` the memory--brilliant! a
"smart pointer" smart pointer already!
    * But sure, the issue is a bit more complicated than this. The above attempt
    only works if one heap memory object has only one pointer pointing to it;
    otherwise other pointers will end up pointing to nowhere, causing
    unexpected behaviors.

## References

* [Microsoft - Smart pointers (Modern C++)](https://learn.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=msvc-170)
* [CPP Reference - smart pointers](https://en.cppreference.com/book/intro/smart_pointers)
* [Standard C++ Foundation - How should I handle resources if my constructors may throw exceptions?](https://isocpp.org/wiki/faq/exceptions#selfcleaning-members)