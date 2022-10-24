# Intel oneMKL

* oneMKL is a part of the `base kit`, if the `base kit` has been installed
following [this link](../), no other installation is needed.

* If the Python wrapper doesn't run, may consider this per [this link](https://github.com/ikinsella/trefide/issues/2): 
```
export LD_PRELOAD=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so
```

* `numpy` usually uses an accelerated linear algebra library to improve performance--typically either Intel MKL or OpenBLAS.
To check which library it is actually using, issue `np.show_config()`

## Results
```
>>> py main.py 
Generating 0.1K random doubles...
mkl_sum():	12.570874509662477,	takes 2.687 ms
np.sum():	12.570874509662477,	takes 0.07176 ms
my_sum():	12.570874509662477,	takes 0.04745 ms

Generating 1.0K random doubles...
mkl_sum():	122.63308019202967,	takes 0.0515 ms
np.sum():	122.63308019202968,	takes 0.05555 ms
my_sum():	122.63308019202967,	takes 0.05007 ms

Generating 10.0K random doubles...
mkl_sum():	1252.8869286717918,	takes 0.05651 ms
np.sum():	1252.8869286717918,	takes 0.06461 ms
my_sum():	1252.8869286717925,	takes 0.05436 ms

Generating 100.0K random doubles...
mkl_sum():	12503.745696040642,	takes 0.1116 ms
np.sum():	12503.745696040636,	takes 0.1185 ms
my_sum():	12503.745696040613,	takes 0.109 ms

Generating 1,000.0K random doubles...
mkl_sum():	124842.96669992828,	takes 0.6111 ms
np.sum():	124842.9666999282,	takes 0.6087 ms
my_sum():	124842.96669992819,	takes 0.5689 ms

Generating 10,000.0K random doubles...
mkl_sum():	1249677.6050475435,	takes 4.795 ms
np.sum():	1249677.605047542,	takes 5.283 ms
my_sum():	1249677.6050475384,	takes 5.078 ms

Generating 100,000.0K random doubles...
mkl_sum():	12500859.451768743,	takes 38.16 ms
np.sum():	12500859.451768894,	takes 40.65 ms
my_sum():	12500859.451768074,	takes 37.75 ms

Generating 1,000,000.0K random doubles...
mkl_sum():	124999729.25400539,	takes 423.8 ms
np.sum():	124999729.25401,	takes 452.6 ms
my_sum():	124999729.25404471,	takes 437.0 ms

Generating 2,000,000.0K random doubles...
mkl_sum():	250000290.83869344,	takes 901.0 ms
np.sum():	250000290.83869845,	takes 947.8 ms
my_sum():	250000290.8387339,	takes 911.4 ms

```