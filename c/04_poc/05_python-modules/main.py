import mylib
import numpy as np
import time


def dummy_func(a):
    return a + 1
  
mylib.helloworld()

start = time.time()

iter_count = 100_000_000
mylib.call_arbitrary_pyfunc(dummy_func, iter_count)
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff} times / sec)')