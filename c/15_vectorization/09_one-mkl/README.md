# Intel oneMKL

* oneMKL is a part of the `base kit`, if the `base kit` has been installed
following [this link](../), no other installation is needed.

* If the Python wrapper doesn't run, may consider this per [this link](https://github.com/ikinsella/trefide/issues/2): 
```
export LD_PRELOAD=/opt/intel/oneapi/mkl/2022.2.0/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/2022.2.0/lib/intel64/libmkl_sequential.so
```

## Results
```
>>> py main.py 
Generating 0.1K random doubles...
mkl_sum():	13.512871281210375,	takes 2.581 ms
np.sum():	13.512871281210373,	takes 0.04005 ms
my_sum():	13.512871281210375,	takes 0.0174 ms

Generating 1.0K random doubles...
mkl_sum():	121.9170148059413,	takes 0.02456 ms
np.sum():	121.91701480594132,	takes 0.01335 ms
my_sum():	121.9170148059413,	takes 0.008106 ms

Generating 10.0K random doubles...
mkl_sum():	1241.2003798826581,	takes 0.009775 ms
np.sum():	1241.200379882658,	takes 0.01359 ms
my_sum():	1241.2003798826577,	takes 0.008345 ms

Generating 100.0K random doubles...
mkl_sum():	12503.530554514951,	takes 0.02885 ms
np.sum():	12503.530554514951,	takes 0.03862 ms
my_sum():	12503.53055451494,	takes 0.04482 ms

Generating 1,000.0K random doubles...
mkl_sum():	124993.24383228763,	takes 0.3374 ms
np.sum():	124993.24383228761,	takes 0.612 ms
my_sum():	124993.24383228837,	takes 0.3448 ms

Generating 10,000.0K random doubles...
mkl_sum():	1249793.0001717184,	takes 4.155 ms
np.sum():	1249793.000171711,	takes 4.472 ms
my_sum():	1249793.000171695,	takes 3.945 ms

Generating 100,000.0K random doubles...
mkl_sum():	12500773.841899103,	takes 41.53 ms
np.sum():	12500773.841899212,	takes 42.48 ms
my_sum():	12500773.841898542,	takes 39.3 ms

Generating 1,000,000.0K random doubles...
mkl_sum():	125002927.25494404,	takes 452.1 ms
np.sum():	125002927.25494047,	takes 462.4 ms
my_sum():	125002927.25496253,	takes 444.1 ms

Generating 2,000,000.0K random doubles...
mkl_sum():	249990484.42186642,	takes 911.6 ms
np.sum():	249990484.421876,	takes 944.4 ms
my_sum():	249990484.42179978,	takes 903.0 ms


```