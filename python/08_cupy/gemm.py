from typing import List

import cupy as cp
import numpy as np
import time


if cp.cuda.is_available():
    print("GPU is available")
else:
    raise RuntimeError('GPU is not available')


m = 30000
k = 8000
n = 11000
alpha = 0.1


def getFloatList(file_path: str, num_read: int) -> List[np.float32]:

    with open(file_path) as f:
        data = []
        count = 0
        for line in f:
            if count >= num_read:
                break
            data.append(np.float32(line.strip()))
            count += 1
    return data


print('Reading A...')
np_A = np.array(getFloatList("./a.in", m * k)).reshape(-1, k, order="F")
assert np_A.dtype == np.float32
print('Done\nReading B...')
np_B = np.array(getFloatList("./b.in", k * n)).reshape(-1, n, order="F")
assert np_B.dtype == np.float32
print('Done')
print("A")
print(np_A)
print("=====")
print("B")
print(np_B)
print("=====")

t0 = time.time()
cp_A = cp.asarray(np_A)
cp_B = cp.asarray(np_B)

t1 = time.time()
dot_product = alpha * cp.dot(cp_A, cp_B)

t2 = time.time()
np_C = cp.asnumpy(dot_product)

t3 = time.time()

print("C")
print(np_C)
print('Writing C...')
np.savetxt("cp.csv.out", np_C, delimiter=",", header='', comments='')
print('Done')
print(f'Moving data to GPU: {(t1 - t0) * 1000:.03f}ms')
print(f'Calculate: {(t2 - t1) * 1000:.03f}ms')
print(f'Moving data to host: {(t3 - t2) * 1000:.03f}ms')
print(f'Total: {(t3 - t0) * 1000:.03f}ms')
