from numpy.ctypeslib import ndpointer

import ctypes
import numpy as np
import numpy.typing as npt
import typing


so_file = "./func.so"
funcs = ctypes.CDLL(so_file)


def sum_an_array(py_list: typing.List[int]) -> int:
    '''
    Here we use the native way, directly from Python list to ctypes array
    '''
    arr_len = len(py_list)
    array_type = ctypes.c_int * arr_len
    arr = array_type(*py_list)
    funcs.sum.restype = ctypes.c_int
    return int(funcs.sum(arr, arr_len))


def bubble_sort_inplace(arr: npt.NDArray[np.int32]) -> None:
    funcs.bubble_sort_inplace.argtypes = [
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        ctypes.c_size_t
    ]
    ret_val = funcs.bubble_sort_inplace(arr, arr.shape[0])
    if (ret_val == 0):
        print(arr)
    else:
        print('C function returns non-zero value!')


print(f'funcs.square(10): {funcs.square(10)}')
print(f'sum_an_array([3, 1, 4, 1, 5]): {sum_an_array([3, 1, 4, 1, 5])}')
bubble_sort_inplace(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))
