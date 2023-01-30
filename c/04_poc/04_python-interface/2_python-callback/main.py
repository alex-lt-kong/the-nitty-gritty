from ctypes import * 
import numpy as np
import time

so_path = r"./out/build/x64-debug/python-callback.dll"
so = CDLL(so_path)

arr_size = 1_000_000
sample_size = 10


@CFUNCTYPE(c_int64, c_uint64)
def py_callback(a):
  return a + 1


def main():

  arr = np.random.randint(0, arr_size, arr_size, dtype=np.uint64)
  sample_idxes = np.random.randint(0, arr_size, sample_size, dtype=np.uint64)

  print(f'{arr_size:,}-element array prepared.\n{sample_size} samples are:   ', end='')
  for i in range(sample_size):
    print(arr[sample_idxes[i]], end=', ')
  print()

  start = time.time()
  so.manipulate_inplace(arr.ctypes.data_as(POINTER(c_uint64)), arr_size, py_callback) 
  diff = time.time() - start

  print(f'{sample_size} samples become:', end='')
  for i in range(sample_size):
    print(arr[sample_idxes[i]], end=', ')
  print()

  print(f'Calling back {arr_size/1_000_000:,.0f}M times takes: {diff:.2f} sec ({arr_size/diff:,.0f} / sec)')


if __name__ == '__main__':
  main()