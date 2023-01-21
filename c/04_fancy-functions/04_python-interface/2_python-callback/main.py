from ctypes import * 
import numpy as np
import time

so_file = "./func.so"
funcs = CDLL(so_file)

arr_size = 2_000_000_000
sample_size = 10;
arr = np.random.randint(0, arr_size, arr_size, dtype=np.int64)
sample_idxes = np.random.randint(0, arr_size, sample_size, dtype=np.int64)
print(f'{arr_size:,}-element array prepared.\n{sample_size} samples are:   ', end='')
for i in range(sample_size):
  print(arr[sample_idxes[i]], end=', ')
print()

CMPFUNC = CFUNCTYPE(c_int, c_int64)
count = 0
def apply(a):
  return a - 1

apply_func = CMPFUNC(apply)

start = time.time()
funcs.manipulate_inplace(arr.ctypes.data_as(POINTER(c_int64)), arr_size, apply_func) 
diff = time.time() - start

print(f'{sample_size} samples become:', end='')
for i in range(sample_size):
  print(arr[sample_idxes[i]], end=', ')
print()

print(f'Calling back {arr_size/1_000_000:,.0f}mil times takes: {diff:.2f} sec ({arr_size/diff:,.0f} / sec)')