from ctypes import *

import ctypes
import datetime as dt
import random
import numpy as np
import pandas as pd
import time


so_file = "./func.so"
func = ctypes.CDLL(so_file)

func.my_multiplication.argtypes = (POINTER(c_double), c_double, c_uint64)
func.my_multiplication.restype = c_double
func.mkl_multiplication.argtypes = (POINTER(c_double), c_double, c_uint64)
func.mkl_multiplication.restype = c_double



arr_sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1000_000_000]

arr = np.ndarray([])
multiplier = 3.1415

for arr_size in arr_sizes:
  print(f"Generating {arr_size / 1_000:,}K random doubles...")
  arr_master = np.random.random((arr_size,)) / 4.0
  sample_idx = random.randint(0, arr_size)  
  
  time.sleep(1)
  arr = arr_master.copy()
  t0 = time.time()
  elpased_ms = func.mkl_multiplication(arr.ctypes.data_as(POINTER(c_double)), multiplier, arr_size)
  t1 = time.time()
  print(f'mkl_multiplication(): arr[{sample_idx}] = {arr[sample_idx]}, takes {(t1 - t0) * 1000:05.4} (ms, per Python) / {elpased_ms:05.4} (ms, per C)')

  time.sleep(1)
  arr = arr_master.copy()
  t0 = time.time()
  arr *= multiplier
  t1 = time.time()
  
  print(f'numpy:                arr[{sample_idx}] = {arr[sample_idx]}, takes {(t1 - t0) * 1000:05.4} (ms, per Python) / NA    (ms, per C)')

  time.sleep(1)
  arr = arr_master.copy()
  t0 = time.time()
  elpased_ms = func.my_multiplication(arr.ctypes.data_as(POINTER(c_double)), multiplier, arr_size)
  t1 = time.time()
  print(f'my_multiplication():  arr[{sample_idx}] = {arr[sample_idx]}, takes {(t1 - t0) * 1000:05.4} (ms, per Python) / {elpased_ms:05.4} (ms, per C)')

  print()