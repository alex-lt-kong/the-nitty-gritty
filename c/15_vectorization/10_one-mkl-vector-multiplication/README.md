# Intel oneMKL

* oneMKL is a part of the `base kit`, if the `base kit` has been installed
following [this link](../), no other installation is needed.

* If the Python wrapper doesn't run, may consider this per [this link](https://github.com/ikinsella/trefide/issues/2): 
```
export LD_PRELOAD=/opt/intel/oneapi/mkl/2022.2.0/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/2022.2.0/lib/intel64/libmkl_sequential.so
```

* Linking options doesn't appear to have a major impact on performance--as long as it still compiles.
  * But Intel provides 
  [an online tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html)
  to tweak them anyway...

## Results
```
>>> py main.py 
Generating 0.1K random doubles...
mkl_multiplication(): arr[85] = 0.7005570546319759, takes 2.609 (ms, per Python) / 2.548 (ms, per C)
numpy:                arr[85] = 0.7005570546319759, takes 0.02098 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[85] = 0.7005570546319759, takes 0.04792 (ms, per Python) / 000.0 (ms, per C)

Generating 1.0K random doubles...
mkl_multiplication(): arr[729] = 0.3957168269944774, takes 0.05007 (ms, per Python) / 0.004 (ms, per C)
numpy:                arr[729] = 0.3957168269944774, takes 0.02074 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[729] = 0.3957168269944774, takes 0.04697 (ms, per Python) / 000.0 (ms, per C)

Generating 10.0K random doubles...
mkl_multiplication(): arr[3224] = 0.5629943121426099, takes 0.06747 (ms, per Python) / 0.019 (ms, per C)
numpy:                arr[3224] = 0.5629943121426099, takes 0.02146 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[3224] = 0.5629943121426099, takes 0.0484 (ms, per Python) / 0.002 (ms, per C)

Generating 100.0K random doubles...
mkl_multiplication(): arr[94283] = 0.28193063376828875, takes 0.08464 (ms, per Python) / 0.036 (ms, per C)
numpy:                arr[94283] = 0.28193063376828875, takes 0.07367 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[94283] = 0.28193063376828875, takes 0.08488 (ms, per Python) / 0.038 (ms, per C)

Generating 1,000.0K random doubles...
mkl_multiplication(): arr[125745] = 0.6200438607587299, takes 0.3829 (ms, per Python) / 0.331 (ms, per C)
numpy:                arr[125745] = 0.6200438607587299, takes 0.8547 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[125745] = 0.6200438607587299, takes 0.4516 (ms, per Python) / 0.385 (ms, per C)

Generating 10,000.0K random doubles...
mkl_multiplication(): arr[3782137] = 0.20311318118993804, takes 7.817 (ms, per Python) / 7.759 (ms, per C)
numpy:                arr[3782137] = 0.20311318118993804, takes 9.785 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[3782137] = 0.20311318118993804, takes 8.036 (ms, per Python) / 7.981 (ms, per C)

Generating 100,000.0K random doubles...
mkl_multiplication(): arr[8704068] = 0.6121153446257397, takes 85.43 (ms, per Python) / 85.37 (ms, per C)
numpy:                arr[8704068] = 0.6121153446257397, takes 095.0 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[8704068] = 0.6121153446257397, takes 86.52 (ms, per Python) / 86.46 (ms, per C)

Generating 1,000,000.0K random doubles...
mkl_multiplication(): arr[139492409] = 0.18940441380876769, takes 815.5 (ms, per Python) / 815.4 (ms, per C)
numpy:                arr[139492409] = 0.18940441380876769, takes 864.3 (ms, per Python) / NA    (ms, per C)
my_multiplication():  arr[139492409] = 0.18940441380876769, takes 839.0 (ms, per Python) / 839.0 (ms, per C)

```