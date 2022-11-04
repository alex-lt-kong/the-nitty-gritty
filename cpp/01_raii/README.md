# Resource acquisition is initialization

* RAII is an awkward name for a useful feature. Essentially, it is equivalent to (if not better than) a `finally` keyword
if properly used

* Let's consider this Python snippet:
```
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