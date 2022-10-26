# * In Python, list is a dynamic array, not a linked list. 
#   By rights, a list in Python should behaves similar to an array in C/C++.
# * However, id(arr[0]) and id(arr[1]) return the same value even though they
#   occupy two different memory blocks. The reason is that, contrary to what we
#   generally expect from C/C++, id() does not return the address of the 1st and 2nd elements.
#   Instead, it returns the value of the 1st and 2nd element, that is, the address of the interger.
# * There is no easy way for people to access the 2nd element in a list by knowing its address according
#   to this now-so-well-received question I asked on Stackoverflow.com:
#   https://stackoverflow.com/questions/66928611/find-the-address-of-the-2nd-element-of-a-list-in-python

import ctypes

def get_by_address(address):
    for x in globals().values():
        if id(x) == address:
            return x


arr = [1, 1, 2, 3, 5, 8, 13]
print(id(arr))
print(id(arr[0]))
print(id(1))
print(id(arr[1]))
print(id(arr[2]))


print(get_by_address(id(arr)))
print(get_by_address(id(arr) + 24))
print(get_by_address(id(arr) + 32))
print(ctypes.cast(id(arr) + 24, ctypes.py_object).value)
# No, this won't work and causes a segmentation fault

