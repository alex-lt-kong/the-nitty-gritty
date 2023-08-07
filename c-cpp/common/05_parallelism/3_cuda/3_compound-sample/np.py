from typing import List

import numpy as np
import time


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
A = np.array(getFloatList("./a.in", m * k)).reshape(-1, k, order="F")
assert A.dtype == np.float32
print(f'Done ({A.size})\nReading B...')
B = np.array(getFloatList("./b.in", k * n)).reshape(-1, n, order="F")
assert B.dtype == np.float32
print(f'Done ({B.size})')
print("A")
print(A)
print("=====")
print("B")
print(B)
print("=====")

t0 = time.time()
A = 0.1 * np.log(A + 11)
C = alpha * np.dot(A, B)
t1 = time.time()

print("C")
print(C)
print('Writing C...')
np.savetxt("np.csv.out", C, delimiter=",", header='', comments='')
print('Done')
print(f'Total: {(t1 - t0) * 1000:.03f}ms')
