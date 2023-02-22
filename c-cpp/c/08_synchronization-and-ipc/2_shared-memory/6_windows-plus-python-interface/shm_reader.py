from numpy.ctypeslib import ndpointer

import ctypes
import numpy as np
import numpy.typing as npt
import time
import typing


so_file = "./shm_reader.so"
shm_reader = ctypes.CDLL(so_file)

shm_reader.read_shm.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
shm_reader.read_shm.restype =  ctypes.c_int
# build python array
arr =  [1,2,3,4,5]

# allocates memory for an equivalent array in C and populates it with
# values from `arr`
arr_c = (ctypes.c_int * 5)(*arr)

retval = shm_reader.read_shm(arr_c, ctypes.c_int(5))

for ele in arr_c:
    print(ele)
