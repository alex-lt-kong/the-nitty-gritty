import datetime as dt
import ctypes
import pandas as pd
import numpy as np
import time

MAX_LINE_COUNT = 32768
CHAR_COL_BUF_SIZE = 64
# These two variables must be the same as those in common.h!

so_file = "./shm_reader.so"
shm_reader = ctypes.CDLL(so_file)

shm_reader.read_shm.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int
]
shm_reader.read_shm.restype =  ctypes.c_int


# allocates memory for an equivalent array in C and populates it with
# values from `arr`
int_arr_c = (ctypes.c_int * MAX_LINE_COUNT)()
dbl_arr_c = (ctypes.c_double * MAX_LINE_COUNT)()
chr_arr_c = (ctypes.c_char * (MAX_LINE_COUNT * CHAR_COL_BUF_SIZE))()

while True:
    
    retval = shm_reader.read_shm(int_arr_c, dbl_arr_c, chr_arr_c, ctypes.c_int(MAX_LINE_COUNT))

    d = {
        'int_col': np.frombuffer(int_arr_c, dtype=np.int32, count=retval),
        'dbl_col': np.frombuffer(dbl_arr_c, dtype=np.double, count=retval),
        'chr_col': np.frombuffer(chr_arr_c[0: retval* CHAR_COL_BUF_SIZE], dtype=f'S{CHAR_COL_BUF_SIZE}')
        # Note that numpy is developed in C as well!
        # np.frombuffer(): Interpret a buffer as a 1-dimensional array.
        # conincidentally, numpy and shm_reader.so organize memory in exactly the same way.

        # In a sense, we have already partially implemented pd.DataFrame in shm_writer.c/shm_reader.c
    }

    df = pd.DataFrame(d)
    df['chr_col'] = df['chr_col'].str.decode("utf-8")
    print(f'[{dt.datetime.now(dt.timezone.utc).timestamp()}] pd.DataFrame()@shm_reader.py returned')
    print(df)
    time.sleep(5)
        