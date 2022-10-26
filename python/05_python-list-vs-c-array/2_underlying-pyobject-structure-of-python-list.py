# In Python, everything is an object, even primitive data types like int.
# In case of a list this will be a PyListObject that has the following definition:
# https://github.com/python/cpython/blob/3.9/Include/cpython/listobject.h#L9-L26

import ctypes
import sys

lst = ["red", "blue", "green"]


class myPyListObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_long),
        ("ob_type", ctypes.c_void_p),
        ("ob_size", ctypes.c_long),
        ("ob_item", ctypes.POINTER(ctypes.c_void_p)),
        ("allocated", ctypes.c_long),
    ]


pylist_obj = myPyListObject.from_address(id(lst))
# from_address: returns a ctypes type instance using the memory specified by address which must be an integer.

for idx, s in enumerate(lst):
    print(f"Enumerating: idx={idx}, s={s}")
    addr = pylist_obj.ob_item[idx]
    s_mem = ctypes.string_at(addr, sys.getsizeof(s))

    print(idx, s_mem[-len(s) - 1 : -1].decode())

