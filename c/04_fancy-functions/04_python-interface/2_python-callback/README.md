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

* Python callback from C
```
1,000,000,000-element array prepared.
10 samples are:    678034294, 387013661, 173291902, 752546974, 334221935, 359489436, 469511013, 622748962, 617584042, 93050447, 
10 samples become: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
Calling back 1,000M times takes: 269.18 sec (3,715,003 / sec)

```

* C callback from C
```
200000000-element array prepared.
10 samples are:    1914701185, 1084009081, 1555864163, 1887325614, 820959790, 224058333, 748474855, 431791477, 1592520960, 1418295238, 
10 samples become: 1914701186, 1084009082, 1555864164, 1887325615, 820959791, 224058334, 748474856, 431791478, 1592520961, 1418295239, 
Calling back 200M times takes: 0.395771 sec (505342.520289K / sec)
```


