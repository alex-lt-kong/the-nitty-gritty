# Coroutine

* Coroutine is a new function in C++20 which gets a lot of hype. It roughly means that two functions (or "routine") can
"co-exist" or works "concurrently" without using multiple threads.

* In the Python world, It is most commonly used the `range()` function, with the help of the `yield` keyword:
```
def rangeN(a, b):
    i = a
    while i < b:
        yield i
        print('Execution resumed@rangeN()')
        i += 1

for i in rangeN(1, 5):
    print(f'i is now {i}')
```

* Coroutine allows a function to "return" a value but still remember its state. When we "call" the same function
again, it resumes the execution since where it left, instead of the beginning.
