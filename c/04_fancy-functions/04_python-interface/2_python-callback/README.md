# C Function Callbacks to a Python Function

* The idea is that we want to first invoke a C function from Python (host
process). One parameter of the C function is a callback function in Python host.
The C function will repetitively call the Python callback function as a
callback.

* So the PoC is a two-step test:
  0. We want to call a C function in a compiled shared object from Python
  host. This Python-to-C function call should pass a function pointer to
  a Python callback function
  0. We want the C function to call the Python callback function repetitively.
  Essentially, it means making a recursively C-to-Python function call.

* It turns out that this is well-supported and documented in Python's official
document [here](https://docs.python.org/3.9/library/ctypes.html#callback-functions).

## Comparison

* While Python callback is not really slow, but C callback is MUCH faster...

* C-to-Python callback
```
1,000,000,000-element array prepared.
10 samples are:    678034294, 387013661, 173291902, 752546974, 334221935, 359489436, 469511013, 622748962, 617584042, 93050447, 
10 samples become: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
Calling back 1,000M times takes: 269.18 sec (3,715,003 / sec)

```

* C-to-C callback
```
1000000000-element array prepared.
10 samples are:    1227037734, 654700958, 1873962047, 1071471667, 2140777588, 1936069344, 648135790, 2063701750, 114406445, 1995835652, 
10 samples become: 1227037735, 654700959, 1873962048, 1071471668, 2140777589, 1936069345, 648135791, 2063701751, 114406446, 1995835653, 
Calling back 1000M times takes: 1.957874 sec (510758.089205K / sec)
```


