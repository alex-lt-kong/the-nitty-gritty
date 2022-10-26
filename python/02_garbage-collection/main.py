# *  Python deletes unwanted objects automatically to free the memory space. The process is run periodically.
# * Official manual for the gc package: https://docs.python.org/3/library/gc.html
# * The following code causes a Segmentation fault exception and Python interpreter will be terminated：


import ctypes
import gc

print('{}\n'.format(gc.get_threshold()))
# Note that the threshold values, which govern the frequency of garbage collection,
# does NOT represent the second or other time units. Instead, they mean the difference
# between the number of object allocations minus the number of object deallocations.
# The larger a value is, the more frequent garbage collection runs

var = 97531
print(var)
var_id = id(var)
print(var_id)
print(id(97531))
print(ctypes.cast(var_id, ctypes.py_object).value)

# Let's manually remove the only reference to 97531 and then call gc.collect()
var = 0
gc.collect()
print(ctypes.cast(var_id, ctypes.py_object).value)
# On Jupyter, it works and 97531 will be printed.
# On native Python, the OS will throw a Segmentation fault exception and kill the Python interpreter.

# This behavior is consistent with the implementation of id() in CPython: if id() returns a
# identification number managed by CPython, which needs to be translated to a memory address,
# then accessing a protected memory block should be detected and blocked by Python itself during 
# translation, so we should expect to get a Python exception.
# However, what we get instead is a OS-level exception, meaning that CPython
# (well at least ctypes.cast() method) does not check the legality of the memory access and directly
# pass the memory access request to the OS, thus getting killed.
# (Note: Sure it is possible that Python does translate the self-defined identification number to
# memory address without verifying the access attempt's legality.)