import mylib
import numpy as np
import time
import custom2


def dummy_func(a):
    if a.number % 10000 == 0:
        #print(a.number)
        pass
    a.number += 1

c = custom2.Custom("asd", 23)
 

mylib.helloworld()

start = time.time()

iter_count = 1_000_000
mylib.call_arbitrary_pyfunc(dummy_func, iter_count, c)
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

