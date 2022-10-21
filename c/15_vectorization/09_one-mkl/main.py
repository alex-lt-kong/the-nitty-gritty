from ctypes import *

import ctypes
import datetime as dt
import numpy as np
import pandas as pd
import time


so_file = "./func.so"
func = ctypes.CDLL(so_file)

func.my_sum.argtypes = [POINTER(c_double), c_uint64]
func.my_sum.restype = c_double
func.mkl_sum.argtypes = [POINTER(c_double), c_uint64]
func.mkl_sum.restype = c_double


arr_sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1000_000_000, 2000_000_000]

arr = np.ndarray([])

for arr_size in arr_sizes:
  print(f"Generating {arr_size / 1_000:,}K random doubles...")
  arr = np.random.random((arr_size,)) / 4.0
  
  t0 = time.time()
  sum = func.mkl_sum(arr.ctypes.data_as(POINTER(c_double)), arr_size)
  t1 = time.time()
  print(f'mkl_sum():\t{sum},\ttakes {(t1 - t0) * 1000:.4} ms')
  
  t0 = time.time()
  sum = np.sum(arr)
  t1 = time.time()
  print(f'np.sum():\t{sum},\ttakes {(t1 - t0) * 1000:.4} ms')

  t0 = time.time()
  sum = func.my_sum(arr.ctypes.data_as(POINTER(c_double)), arr_size)
  t1 = time.time()
  print(f'my_sum():\t{sum},\ttakes {(t1 - t0) * 1000:.4} ms')

  print()