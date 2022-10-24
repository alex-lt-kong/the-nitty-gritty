from ctypes import POINTER, c_double, c_int64, c_uint64, byref

import ctypes
import datetime as dt
import random
import numpy as np
import pandas as pd
import time


so_file = "./func.so"
func = ctypes.CDLL(so_file)

c_product = c_double(0)
func.my_dot_product.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int64)
func.my_dot_product.restype = c_double
func.mkl_dot_product.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int64)
func.mkl_dot_product.restype = c_double

arr_sizes = np.array([100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 200_000_000, 300_000_000], dtype=np.int64)

for arr_size in arr_sizes:
  print(f"Generating two vectors, each with {arr_size / 1_000:,}K random doubles...")
  vec_a_master = np.random.random((arr_size,))
  vec_b_master = np.random.random((arr_size,))

  time.sleep(1)
  vec_a = vec_a_master.copy()
  vec_b = vec_b_master.copy()
  t0 = time.time()
  product = np.dot(vec_a, vec_b)
  t1 = time.time()
  print(
      f'numpy:             product = {product:,.5f}, '
      f'takes {(t1 - t0) * 1000:,.2f} (ms, per Python) / NA    (ms, per C)'
  )

  time.sleep(1)
  vec_a = vec_a_master.copy()
  vec_b = vec_b_master.copy()
  t0 = time.time()
  elpased_ms = func.mkl_dot_product(
      vec_a.ctypes.data_as(POINTER(c_double)), vec_b.ctypes.data_as(POINTER(c_double)), byref(c_product), arr_size
  )
  t1 = time.time()
  print(
      f'mkl_dot_product(): product = {c_product.value:,.5f}, '
      f'takes {(t1 - t0) * 1000:,.2f} (ms, per Python) / {elpased_ms:,.2f} (ms, per C)'
  )

  time.sleep(1)
  vec_a = vec_a_master.copy()
  vec_b = vec_b_master.copy()
  t0 = time.time()
  elpased_ms = func.my_dot_product(
      vec_a.ctypes.data_as(POINTER(c_double)), vec_b.ctypes.data_as(POINTER(c_double)), byref(c_product), arr_size
  )
  t1 = time.time()
  print(
      f'my_dot_product():  product = {c_product.value:,.5f}, '
      f'takes {(t1 - t0) * 1000:,.2f} (ms, per Python) / {elpased_ms:,.2f} (ms, per C)'
  )


  print()
