# C Function Callbacks to a Python Function

* The idea is that we need to first invoke a C function from Python host. One
parameter of the C function is a callback function in Python. The C function
will repetitively call the Python as a callback.

```
2,000,000,000-element array prepared.
10 samples are:   646935167, 1516949280, 1160166857, 788043348, 670999919, 241956719, 1516338654, 17793485, 1198180916, 43358535, 
10 samples become:646935166, 1516949279, 1160166856, 788043347, 670999918, 241956718, 1516338653, 17793484, 1198180915, 43358534, 
Calling back 2,000M times takes: 363.06 sec (5,508,787 / sec)
```

```
200000000-element array prepared.
10 samples are:    1914701185, 1084009081, 1555864163, 1887325614, 820959790, 224058333, 748474855, 431791477, 1592520960, 1418295238, 
10 samples become: 1914701186, 1084009082, 1555864164, 1887325615, 820959791, 224058334, 748474856, 431791478, 1592520961, 1418295239, 
Calling back 200M times takes: 0.395771 sec (505342.520289K / sec)
```

https://docs.python.org/3.3/library/ctypes.html?highlight=ctypes#callback-functions
