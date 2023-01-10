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

    * To be specific, RAII encapsulates each resource into a class, where:
        * the constructor acquires the resource and establishes all class
        invariants or throws an exception if that cannot be done,
        * the destructor releases the resource and never throws exceptions; 


* Similar to Python's Exception raised in `finally` section, if exception
is thrown within destructors, it has to be handled manually. This is beyond
the scope of RAII.