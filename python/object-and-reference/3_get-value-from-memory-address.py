# You can get a dictionary of the current global symbol table by calling globals().
# You can get the value of a variable from its memory address by either:
# * looking at the globals() dictionary
# * using ctypes's cast() method. ctypes is a foreign function library for Python.

a = 103
b = 103
a += 1
a -= 1
a += 1

print(globals())

import ctypes

address = id(a)
print(address)

def get_by_address(address):
    for x in globals().values():
        if id(x) == address:
            return x

print(get_by_address(address=address))

print(ctypes.cast(address, ctypes.py_object).value)

