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

fp = open('quicksort.in', 'r')
arr = []
lines = fp.readlines()
for line in lines:
  arr.append(int(line))
fp.close()

arr_copy = arr.copy()

start = time.time()
quick_sort(arr_copy, 0, len(arr_copy) - 1)
print(f"Pure python: {time.time() - start:.3f} sec")
fp = open('quicksort.out.py-pure', 'w')
for i in range(len(arr_copy)):
  fp.write(str(arr_copy[i]) + ", ")
fp.close()

arr_copy = arr.copy()
start = time.time()
arr_copy.sort()
print(f"list.sort(): {time.time() - start:.3f} sec")
fp = open('quicksort.out.py-stdlib', 'w')
for i in range(len(arr_copy)):
  fp.write(str(arr_copy[i]) + ", ")
fp.close()

arr_copy = np.array(arr.copy())
start = time.time()
arr_copy = np.sort(arr_copy)
print(f"Numpy.sort(): {time.time() - start:.3f} sec")
fp = open('quicksort.out.py-numpy', 'w')
for i in range(len(arr_copy)):
  fp.write(str(arr_copy[i]) + ", ")
fp.close()