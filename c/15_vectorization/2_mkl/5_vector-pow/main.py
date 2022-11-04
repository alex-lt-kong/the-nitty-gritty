from ctypes import POINTER, c_double, c_uint64

import ctypes
import random
import numpy as np
import time


so_file = "./func.so"
func = ctypes.CDLL(so_file)

c_product = c_double(0)
func.my_pow.argtypes = (c_uint64, POINTER(c_double), POINTER(c_double), POINTER(c_double))
func.my_pow.restype = c_double
func.mkl_pow.argtypes = (c_uint64, POINTER(c_double), POINTER(c_double), POINTER(c_double))
func.mkl_pow.restype = c_double

arr_sizes = np.array(
    [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 200_000_000, 400_000_000],
    dtype=np.int64
)

for arr_size in arr_sizes:
  print(f"Generating vector vec_in with {arr_size / 1_000:,}K random doubles...")
  vec_base = np.random.random((arr_size,))
  vec_exp = np.random.random((arr_size,))
  rnd_idx = random.randint(0, arr_size - 1)

  time.sleep(1)
  vec_out = np.empty((arr_size, ), dtype=np.double)
  t0 = time.time()
  vec_out = np.power(vec_base, vec_exp)
  t1 = time.time()
  print(
      f'numpy:     vec_out[{rnd_idx:,}] = {vec_base[rnd_idx]:,.5f}^{vec_exp[rnd_idx]:,.5f} = {vec_out[rnd_idx]:,.5f}, '
      f'takes {(t1 - t0) * 1000:8,.2f} (ms, per Python) /       NA (ms, per C)'
  )

  time.sleep(1)
  vec_out = np.empty((arr_size, ), dtype=np.double)
  t0 = time.time()
  elpased_ms = func.mkl_pow(
      arr_size,
      vec_base.ctypes.data_as(POINTER(c_double)),
      vec_exp.ctypes.data_as(POINTER(c_double)),
      vec_out.ctypes.data_as(POINTER(c_double))
  )
  t1 = time.time()
  print(
      f'mkl_pow(): vec_out[{rnd_idx:,}] = {vec_base[rnd_idx]:,.5f}^{vec_exp[rnd_idx]:,.5f} = {vec_out[rnd_idx]:,.5f}, '
      f'takes {(t1 - t0) * 1000:8,.2f} (ms, per Python) / {elpased_ms:8,.2f} (ms, per C)'
  )

  time.sleep(1)
  vec_out = np.empty((arr_size, ), dtype=np.double)
  t0 = time.time()
  elpased_ms = func.my_pow(
      arr_size,
      vec_base.ctypes.data_as(POINTER(c_double)),
      vec_exp.ctypes.data_as(POINTER(c_double)),
      vec_out.ctypes.data_as(POINTER(c_double))
  )
  t1 = time.time()
  print(
      f'my_ln():   vec_out[{rnd_idx:,}] = {vec_base[rnd_idx]:,.5f}^{vec_exp[rnd_idx]:,.5f} = {vec_out[rnd_idx]:,.5f}, '
      f'takes {(t1 - t0) * 1000:8,.2f} (ms, per Python) / {elpased_ms:8,.2f} (ms, per C)'
  )

  print()
