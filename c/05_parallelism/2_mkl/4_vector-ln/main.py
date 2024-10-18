from ctypes import POINTER, c_double, c_uint64

import ctypes
import random
import numpy as np
import sys
import time


if sys.platform == 'win32':
  lib_file = "./build/Release/func.dll"
else:
  lib_file = "./build/libfunc.so"
func = ctypes.CDLL(lib_file)


c_product = c_double(0)
func.my_ln.argtypes = (c_uint64, POINTER(c_double), POINTER(c_double))
func.my_ln.restype = c_double
func.mkl_ln.argtypes = (c_uint64, POINTER(c_double), POINTER(c_double))
func.mkl_ln.restype = c_double

arr_sizes = np.array(
    [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 200_000_000, 400_000_000],
    dtype=np.int64
)

for arr_size in arr_sizes:
  print(f"Generating vector vec_in with {arr_size / 1_000:,}K random doubles...")
  vec_in = np.random.random((arr_size,))
  rnd_idx = random.randint(0, arr_size - 1)

  time.sleep(1)
  vec_out = np.empty((arr_size, ), dtype=np.double)
  t0 = time.time()
  vec_out = np.log(vec_in)
  t1 = time.time()
  print(
      f'numpy:    vec_out[{rnd_idx:,}] = ln({vec_in[rnd_idx]:,.5f}) = {vec_out[rnd_idx]:,.5f}, '
      f'takes {(t1 - t0) * 1000:8,.2f} (ms, per Python) /       NA (ms, per C)'
  )

  time.sleep(1)
  vec_out = np.empty((arr_size, ), dtype=np.double)
  t0 = time.time()
  elpased_ms = func.my_ln(
      arr_size, vec_in.ctypes.data_as(POINTER(c_double)), vec_out.ctypes.data_as(POINTER(c_double))
  )
  t1 = time.time()
  print(
      f'my_ln():  vec_out[{rnd_idx:,}] = ln({vec_in[rnd_idx]:,.5f}) = {vec_out[rnd_idx]:,.5f}, '
      f'takes {(t1 - t0) * 1000:8,.2f} (ms, per Python) / {elpased_ms:8,.2f} (ms, per C)'
  )

  time.sleep(1)
  vec_out = np.empty((arr_size, ), dtype=np.double)
  t0 = time.time()
  elpased_ms = func.mkl_ln(
      arr_size, vec_in.ctypes.data_as(POINTER(c_double)), vec_out.ctypes.data_as(POINTER(c_double))
  )
  t1 = time.time()
  print(
      f'mkl_ln(): vec_out[{rnd_idx:,}] = ln({vec_in[rnd_idx]:,.5f}) = {vec_out[rnd_idx]:,.5f}, '
      f'takes {(t1 - t0) * 1000:8,.2f} (ms, per Python) / {elpased_ms:8,.2f} (ms, per C)'
  )

  print()
