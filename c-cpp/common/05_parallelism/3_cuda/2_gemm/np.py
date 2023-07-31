from typing import List

import numpy as np
import time


m = 30000
k = 8000
n = 1100
alpha = 0.1


def getDoubleList(file_path: str, num_read: int) -> List[np.float32]:

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
A = np.array(getDoubleList("./a.in", m * k)).reshape(-1, k, order="F")
assert A.dtype == np.float32
print('Done\nReading B...')
B = np.array(getDoubleList("./b.in", k * n)).reshape(-1, n, order="F")
assert B.dtype == np.float32
print('Done')
print("A")
print(A)
print("=====")
print("B")
print(B)
print("=====")

t0 = time.time()
C = alpha * np.dot(A, B)
t1 = time.time()

print("C")
print(C)
print('Writing C...')
np.savetxt("np.csv.out", C, delimiter=",", header='', comments='')
print('Done')
print(f'Total: {(t1 - t0) * 1000:.03f}ms')
