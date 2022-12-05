import numpy as np
import numpy.typing as npt
import typing


def insertion_sort_inplace(arr: npt.NDArray[np.int32], arr_offset: int, arr_len: int) -> None:
  for i in range(arr_offset, arr_offset + arr_len):
    key = arr[i]
    j = i - 1

    while key < arr[j] and j >= 0:
      arr[j + 1] = arr[j]
      j -= 1
    
    arr[j + 1] = key


def pick_pivot_index(arr: npt.NDArray[np.int32], lo: int, hi: int) -> int:
    if hi - lo <= 2:
        return lo

    mid = int((lo + hi) / 2)
    if (arr[lo] <= arr[mid] and arr[mid] <= arr[hi]) or (arr[lo] >= arr[mid] and arr[mid] >= arr[hi]):
        return mid
    if (arr[mid] <= arr[lo] and arr[lo] <= arr[hi]) or (arr[mid] >= arr[lo] and arr[lo] >= arr[hi]):
        return lo
    return hi


def partition(arr: npt.NDArray[np.int32], lo: int, hi: int) -> int:
    pos = lo - 1
    pivot_idx = pick_pivot_index(arr, lo, hi)
    t = arr[pivot_idx]
    arr[pivot_idx] = arr[hi]
    arr[hi] = t
    pivot = arr[hi]
    for i in range(lo, hi+1):
        if (arr[i] < pivot):
            pos += 1
            t = arr[i]
            arr[i] = arr[pos]
            arr[pos] = t
    pos += 1
    t = arr[hi]
    arr[hi] = arr[pos]
    arr[pos] = t
    return pos


def quick_sort_inplace(arr: npt.NDArray[np.int32], lo: int, hi: int) -> None:
    thres = 10
    if hi - lo > thres:
        pos = partition(arr, lo, hi)
        quick_sort_inplace(arr, lo, pos - 1)
        quick_sort_inplace(arr, pos + 1, hi)
    else:
        # The idea is that, if the array is too small, we fallback to insertion sort to boost performance.
        insertion_sort_inplace(arr, lo, hi - lo + 1)
