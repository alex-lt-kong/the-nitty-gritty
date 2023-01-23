# C Function Callbacks to a Python Function

* The idea is that we want to first invoke a C function from Python (host
process). One parameter of the C function is a function pointer pointing
to a function in Python host (Python callback). The C function will then
repetitively call the Python callback function using the function pointer.

* So the PoC is a two-step test:

  0. We want to call a C function in a compiled shared object from Python
  host. This Python-to-C function call should pass a function pointer to
  a Python callback function.
  0. We want the C function to call the Python callback function in a loop.
  Essentially, it means making a recursive C-to-Python function call.

* It turns out that this is well-supported and documented in Python's official
document [here](https://docs.python.org/3.9/library/ctypes.html#callback-functions).

## Comparison

* While C-to-Python callback is not really slow, but C-to-C callback is 
MUCH faster...

* C-to-Python callback
```
1,000,000,000-element array prepared.
10 samples are:   114303050, 571202881, 38925011, 288658365, 410043202, 596894078, 360248691, 256052724, 499385069, 197724980, 
10 samples become:114303049, 571202880, 38925010, 288658364, 410043201, 596894077, 360248690, 256052723, 499385068, 197724979, 
Calling back 1,000M times takes: 166.26 sec (6,014,744 / sec)
```

* C-to-C callback
```
1000000000-element array prepared.
10 samples are:    1227037734, 654700958, 1873962047, 1071471667, 2140777588, 1936069344, 648135790, 2063701750, 114406445, 1995835652, 
10 samples become: 1227037735, 654700959, 1873962048, 1071471668, 2140777589, 1936069345, 648135791, 2063701751, 114406446, 1995835653, 
Calling back 1000M times takes: 1.957874 sec (510758.089205K / sec)
```


