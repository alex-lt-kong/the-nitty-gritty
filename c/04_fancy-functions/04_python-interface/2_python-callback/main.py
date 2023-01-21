from ctypes import * 
import numpy as np
from numpy.ctypeslib import ndpointer

so_file = "./func.so"
funcs = CDLL(so_file)


IntArray5 = c_int * 5
ia = IntArray5(5, 1, 7, 33, 99)
qsort = funcs.qsort
qsort.restype = None

CMPFUNC = CFUNCTYPE(c_int, POINTER(c_int), POINTER(c_int))
def py_cmp_func(a, b):
  print("py_cmp_func", a[0], b[0])
  return a[0] > b[0]

cmp_func = CMPFUNC(py_cmp_func)

funcs.qsort(ia, len(ia), sizeof(c_int), cmp_func) 

for i in range(5):
  print(ia[i])
