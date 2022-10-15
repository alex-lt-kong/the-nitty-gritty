from numpy.ctypeslib import ndpointer

import ctypes
import pandas as pd
import numpy as np
import time

so_file = "./shm_reader.so"
shm_reader = ctypes.CDLL(so_file)

shm_reader.read_shm.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_char), ctypes.c_int]
shm_reader.read_shm.restype =  ctypes.c_int
# build python array
arr =  [1,2,3,4,5]
str_arr = [35]

# allocates memory for an equivalent array in C and populates it with
# values from `arr`
int_arr_c = (ctypes.c_int * 128)(*arr)
dbl_arr_c = (ctypes.c_double * 128)(*arr)
chr_arr_c = (ctypes.c_char * (128 * 128))(*str_arr)

while True:
    time.sleep(10)
    retval = shm_reader.read_shm(int_arr_c, dbl_arr_c, chr_arr_c, ctypes.c_int(128))

    #for i in range(retval):
    #    print(int_arr_c[i])
    #for i in range(retval):
    #    print(dbl_arr_c[i])

    str_col = [128] * retval

    for i in range(retval):
        str_col[i] = chr_arr_c[i * 128 : (i+1) * 128].decode('utf-8').rstrip('\0')
    #    print(str_col[i])
    #print(str_col)

    d = {
        'int_col': np.frombuffer(int_arr_c, dtype=np.int32, count=retval),
        'dbl_col': np.frombuffer(dbl_arr_c, dtype=np.double, count=retval),
        'chr_col': np.array(str_col)
    }

    df = pd.DataFrame(d)
    print(df)
        