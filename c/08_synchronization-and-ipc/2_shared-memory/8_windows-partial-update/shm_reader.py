import datetime as dt
import ctypes
import pandas as pd
import numpy as np
import time

MAX_LINE_COUNT = 32768
CHAR_COL_BUF_SIZE = 256
# These two variables must be the same as those in common.h!

so_file = "./shm_reader.so"
shm_reader = ctypes.CDLL(so_file)

shm_reader.read_shm.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_char), ctypes.c_int]
shm_reader.read_shm.restype =  ctypes.c_int
# build python array
arr =  [0]
dt_arr = [35]
str_arr = [35]

# allocates memory for an equivalent array in C and populates it with
# values from `arr`
int_arr_c = (ctypes.c_int * MAX_LINE_COUNT)(*arr)
dt_arr_c  = (ctypes.c_char * (MAX_LINE_COUNT * CHAR_COL_BUF_SIZE))(*dt_arr)
dbl_arr_c = (ctypes.c_double * MAX_LINE_COUNT)(*arr)
chr_arr_c = (ctypes.c_char * (MAX_LINE_COUNT * CHAR_COL_BUF_SIZE))(*str_arr)

while True:
    
    retval = shm_reader.read_shm(int_arr_c, dt_arr_c, dbl_arr_c, chr_arr_c, ctypes.c_int(MAX_LINE_COUNT))
    str_col = [CHAR_COL_BUF_SIZE] * retval
    dt_col = [CHAR_COL_BUF_SIZE] * retval
    for i in range(retval):
        dt_col[i] = dt_arr_c[i * CHAR_COL_BUF_SIZE : (i+1) * CHAR_COL_BUF_SIZE].split(b'\x00')[0].decode('utf-8')
        str_col[i] = chr_arr_c[i * CHAR_COL_BUF_SIZE : (i+1) * CHAR_COL_BUF_SIZE].split(b'\x00')[0].decode('utf-8')
        

    d = {
        'int_col': np.frombuffer(int_arr_c, dtype=np.int32, count=retval),
        'dt_col': np.array(dt_col),
        'dbl_col': np.frombuffer(dbl_arr_c, dtype=np.double, count=retval),
        'chr_col': np.array(str_col)
    }

    df = pd.DataFrame(d)
    print(f'[{dt.datetime.now(dt.timezone.utc).timestamp()}] pd.DataFrame()@shm_reader.py returned')
    print(df)
   # df.to_csv('C:\\temp.csv')
    time.sleep(5)
        