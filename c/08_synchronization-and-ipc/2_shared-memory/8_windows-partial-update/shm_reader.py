from ctypes import *

import ctypes
import datetime as dt
import numpy as np
import pandas as pd
import time


MAX_LINE_COUNT = 65536 # 2 ** 16
CHAR_COL_BUF_SIZE = 256
# These two variables must be the same as those in common.h!

so_file = "./shm_reader.so"
shm_reader = ctypes.CDLL(so_file)

shm_reader.read_shm.argtypes = [
    POINTER(c_int),
    POINTER(c_char),
    POINTER(c_double),
    POINTER(c_char),
    c_uint64, c_uint64]
shm_reader.read_shm.restype =  c_int

# allocates memory for an equivalent array in C
int_arr_c = (ctypes.c_int * MAX_LINE_COUNT)()
dt_arr_c  = (ctypes.c_char * (MAX_LINE_COUNT * CHAR_COL_BUF_SIZE))()
dbl_arr_c = (ctypes.c_double * MAX_LINE_COUNT)()
chr_arr_c = (ctypes.c_char * (MAX_LINE_COUNT * CHAR_COL_BUF_SIZE))()

hi_los = [    
    [0, 0],
    [1, 1],
    [1, 0],
    [MAX_LINE_COUNT - 1, MAX_LINE_COUNT - 10],
    [16584, 16384],
    [5432, 2048],
    [MAX_LINE_COUNT - 1000, MAX_LINE_COUNT - 11000],
    [MAX_LINE_COUNT - 1, 0]
]
idx = 0

while True:
    time.sleep(1)
    hi = hi_los[idx][0]
    lo = hi_los[idx][1]
    idx += 1
    if idx >= len(hi_los):
        idx = 0
    print(
        f'[{dt.datetime.now(dt.timezone.utc).timestamp()}] calling read_shm()@shm_reader.so with [{hi}, {lo}], length: {hi - lo + 1}'
    )
    retval = shm_reader.read_shm(
        int_arr_c, dt_arr_c, dbl_arr_c, chr_arr_c, c_uint64(hi), c_uint64(lo)
    )
    if retval == 0: # most likely internal error
        continue
    d = {
        'int_col': np.frombuffer(int_arr_c, dtype=np.int32, count=retval),
        'dt_col': np.frombuffer(dt_arr_c[0: retval* CHAR_COL_BUF_SIZE], dtype=f'S{CHAR_COL_BUF_SIZE}'),
        'dbl_col': np.frombuffer(dbl_arr_c, dtype=np.double, count=retval),
        'chr_col': np.frombuffer(chr_arr_c[0: retval* CHAR_COL_BUF_SIZE], dtype=f'S{CHAR_COL_BUF_SIZE}')
    }

    df = pd.DataFrame(d)
    df['dt_col'] = df['dt_col'].str.decode("utf-8")
    df['chr_col'] = df['chr_col'].str.decode("utf-8")
    print(f'[{dt.datetime.now(dt.timezone.utc).timestamp()}] pd.DataFrame()@shm_reader.py returned')
    print(df)
    print('')
        