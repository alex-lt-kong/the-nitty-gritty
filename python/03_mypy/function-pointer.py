from typing import Callable

def add(a: int, b: int) -> int:
  return a + b

def minus(a: int, b: int) -> int:
  return a - b

def number_manipulation(func: Callable[[int, int], int], a: int, b: int) -> int:
  return func(a, b)

print(number_manipulation(add, 1, 2))
print(number_manipulation(minus, 1, 2))