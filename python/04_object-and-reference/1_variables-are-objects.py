# The default Python implementation, CPython, is actually written in the C programming language. Let's check out whether we are using CPython first!

import platform
print(platform.python_implementation())

# one should be aware that the id() of an object is implementation-defined. CPython implementation returns the address of the object in memory when you call id() function.
# It is in our expectation that one can check the address of a "variable", such as an integer.

a = 1233.45
b = a
 
print(id(a))
print(id(b))