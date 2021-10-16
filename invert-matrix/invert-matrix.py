import numpy as np
import time

mat = np.random.randint(low=0, high=65536, size=(200, 200))

print('Normal inverse')
start = time.time()
for i in range(1000):
  np.linalg.inv(mat)
print(f"{(time.time() - start):.1f} ms")

print('Moore–Penrose inverse')
start = time.time()
for i in range(1000):
  np.linalg.pinv(mat)
print(f"{(time.time() - start):.1f} ms")

