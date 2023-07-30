from typing import List

import numpy as np
import time


m = 100 * 1000
k = 1999
n = 30


def getDoubleList(file_path: str, num_read: int) -> List[float]:

    with open(file_path) as f:
        data = []
        count = 0
        for line in f:
            if count >= num_read:
                break
            data.append(float(line.strip()))
            count += 1
    return data


print('Reading A...')
A = np.array(getDoubleList("./a.in", m * k)).reshape(-1, k, order="F")
print('Done\nReading B...')
B = np.array(getDoubleList("./b.in", k * n)).reshape(-1, n, order="F")
print('Done')

print("A")
print(A)
print("=====")
print("B")
print(B)
print("=====")

t0 = time.time()
A = np.log(A + 11)
A *= 0.1
C = np.dot(A, B)
t1 = time.time()

print("C")
print(C)

print(f'{(t1 - t0) * 1000}ms')
