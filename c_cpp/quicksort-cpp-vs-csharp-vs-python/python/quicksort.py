import random
import time
import numpy as np

def partition(arr, li, hi):
    index = li
    pi = random.randrange(li, hi + 1)
    arr[hi], arr[pi] = arr[pi], arr[hi]
    pivot = arr[hi]
  
    for j in range(li, hi):
        if arr[j] <= pivot:            
            arr[index], arr[j] = arr[j], arr[index]
            index = index + 1
            
    arr[index], arr[hi] = arr[hi], arr[index]
    return (index)
  
def quick_sort(arr, li, hi):
    if len(arr) == 1:
        return arr
    
    if li < hi:        
        pi = partition(arr, li, hi)
        quick_sort(arr, li, pi-1)
        quick_sort(arr, pi+1, hi)

def read_numbers(i):
  fp = open(f'quicksort.in{i}', 'r')
  arr = []
  lines = fp.readlines()
  for line in lines:
    arr.append(int(line))
  fp.close()
  return arr

print('Pure Python:')
iter_count = 10
average_elapsed_ms = 0
for i in range(iter_count):  
  arr = read_numbers(i)

  start = time.time()
  quick_sort(arr, 0, len(arr) - 1)
  end = time.time()
  average_elapsed_ms += (end - start)
  print(f"{i}-th iteration: {(end - start) * 1000:,.0f} ms")
  fp = open(f'quicksort.py.out{i}', 'w')
  for i in range(len(arr)):
    fp.write(str(arr[i]) + ", ")
  fp.close()
print(f"Average: {average_elapsed_ms * 1000 / iter_count:,.0f}ms")

print('list.sort():')
average_elapsed_ms = 0
for i in range(iter_count):
  arr = read_numbers(i)
  start = time.time()
  arr.sort()
  end = time.time()
  average_elapsed_ms += (end - start)
  print(f"{i}-th iteration: {(end - start) * 1000:,.0f} ms")
  fp = open(f'quicksort.list-sort.out{i}', 'w')
  for i in range(len(arr)):
    fp.write(str(arr[i]) + ", ")
  fp.close()
print(f"Average: {average_elapsed_ms * 1000 / iter_count:,.0f}ms")

print('numpy.sort():')
average_elapsed_ms = 0
for i in range(iter_count):
  arr = read_numbers(i)
  start = time.time()
  arr = np.sort(arr)
  end = time.time()
  average_elapsed_ms += (end - start)
  print(f"{i}-th iteration: {(end - start) * 1000:,.0f} ms")
  fp = open(f'quicksort.numpy-sort.out{i}', 'w')
  for i in range(len(arr)):
    fp.write(str(arr[i]) + ", ")
  fp.close()
print(f"Average: {average_elapsed_ms * 1000 / iter_count:,.0f}ms")