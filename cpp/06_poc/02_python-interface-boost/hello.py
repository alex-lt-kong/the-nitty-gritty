import time
import hello 
import numpy as np

limit = 5_000_000

print('Numpy vectorization')
start = time.time()
print(np.sum(np.arange(limit)))
print(f'{time.time() - start:.6} sec')

print()

print('C++ naive loop')
start = time.time()
hello.sum_a_list(limit)
print(f'{time.time() - start:.6} sec')