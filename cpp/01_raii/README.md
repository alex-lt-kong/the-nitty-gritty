# Resource acquisition is initialization

* RAII is an awkward name for a useful feature. Essentially, it is equivalent to (if not better than) a `finally` keyword
if properly used

* Let's consider this Python snippet:
    ```python
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        # db operations
    except Exception as ex:
        pass
    finally:
        if conn is not None:
            conn.close()
    ```

    * `finally` is needed because we don't want `conn` to be left open whether or not an exception is thrown.
    * This is also the "official" opinion of [Bjarne Stroustrup](https://www.stroustrup.com/bs_faq2.html#finally)

* C++'s answer is rather straightforward, let's wrap the resource releasing in destructor and always call destructor
even if an exception is thrown:
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