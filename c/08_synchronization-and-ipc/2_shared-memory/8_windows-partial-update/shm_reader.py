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
    POINTER(POINTER(c_int)),
    POINTER(POINTER(c_char)),
    POINTER(POINTER(c_double)),
    POINTER(POINTER(c_char)),
    c_uint64, c_uint64]
shm_reader.read_shm.restype =  c_int

int_ptr =  POINTER(c_int)()
dt_ptr = POINTER(c_char)()
dbl_ptr =  POINTER(c_double)()
chr_ptr = POINTER(c_char)()

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
    time.sleep(5)
    hi = hi_los[idx][0]
    lo = hi_los[idx][1]
    idx += 1
    if idx >= len(hi_los):
        idx = 0
    print(
        f'[{dt.datetime.now(dt.timezone.utc).timestamp()}] calling read_shm()@shm_reader.so from shm_reader.py with [{hi}, {lo}], length: {hi - lo + 1}'
    )
    retval = shm_reader.read_shm(
        byref(int_ptr), byref(dt_ptr), byref(dbl_ptr), byref(chr_ptr), c_uint64(hi), c_uint64(lo)
    )
    if retval == 0: # most likely internal error
        continue
    d = {
        'int_col':  np.ctypeslib.as_array(int_ptr, shape=(retval,)),
        'dt_col': np.frombuffer(dt_ptr[0: retval* CHAR_COL_BUF_SIZE], dtype=f'S{CHAR_COL_BUF_SIZE}'),
        'dbl_col': np.ctypeslib.as_array(dbl_ptr, shape=(retval,)),
        'chr_col': np.frombuffer(chr_ptr[0: retval* CHAR_COL_BUF_SIZE], dtype=f'S{CHAR_COL_BUF_SIZE}')
    }

    df = pd.DataFrame(d)
    print(f'[{dt.datetime.now(dt.timezone.utc).timestamp()}] pd.DataFrame()@shm_reader.py returned')
    print(df)
    print('')
        