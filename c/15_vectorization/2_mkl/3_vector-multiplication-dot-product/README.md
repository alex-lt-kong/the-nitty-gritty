# Intel oneMKL

* If the Python wrapper doesn't run and complains `INTEL MKL ERROR: ...libmkl_avx2.so.2: undefined symbol: mkl_sparse_optimize_bsr_trsm_i8`,
may consider this per [this link](https://github.com/ikinsella/trefide/issues/2): 
```
export LD_PRELOAD=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so
```
* Linking options doesn't appear to have a major impact on performance--as long as it still compiles.
  * But Intel provides 
  [an online tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html)
  to tweak them anyway...

* `numpy` usually uses an accelerated linear algebra library to improve performance--typically either Intel MKL or OpenBLAS.
To check which library it is actually using, issue `np.show_config()`

* It is very likely that the dot product calculation is memory-bound--given the size of the data and the predictable
pattern, prefetchers/caching should have already been fully operational. Perhaps we are
approaching the hard limit of memory bandwidth.
  * A useful [reference](https://stackoverflow.com/questions/18159455/why-vectorizing-the-loop-does-not-have-performance-improvement/18159503#18159503)

## Results
```
>>> py main.py 
Generating two vectors, each with 0.1K random doubles...
numpy:             product = 25.92156, takes 0.05 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 25.92156, takes 2.64 (ms, per Python) / 2.57 (ms, per C)
my_dot_product():  product = 25.92156, takes 0.09 (ms, per Python) / 0.00 (ms, per C)

Generating two vectors, each with 1.0K random doubles...
numpy:             product = 243.82504, takes 0.02 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 243.82504, takes 0.06 (ms, per Python) / 0.00 (ms, per C)
my_dot_product():  product = 243.82504, takes 0.05 (ms, per Python) / 0.00 (ms, per C)

Generating two vectors, each with 10.0K random doubles...
numpy:             product = 2,463.13575, takes 0.03 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 2,463.13575, takes 0.06 (ms, per Python) / 0.01 (ms, per C)
my_dot_product():  product = 2,463.13575, takes 0.06 (ms, per Python) / 0.00 (ms, per C)

Generating two vectors, each with 100.0K random doubles...
numpy:             product = 25,061.59725, takes 0.11 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 25,061.59725, takes 0.10 (ms, per Python) / 0.05 (ms, per C)
my_dot_product():  product = 12,476.10601, takes 0.18 (ms, per Python) / 0.13 (ms, per C)

Generating two vectors, each with 1,000.0K random doubles...
numpy:             product = 250,247.87103, takes 8.21 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 250,247.87103, takes 0.66 (ms, per Python) / 0.61 (ms, per C)
my_dot_product():  product = 125,036.85575, takes 0.56 (ms, per Python) / 0.47 (ms, per C)

Generating two vectors, each with 10,000.0K random doubles...
numpy:             product = 2,500,641.08617, takes 8.51 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 2,500,641.08617, takes 8.72 (ms, per Python) / 8.66 (ms, per C)
my_dot_product():  product = 2,500,641.08617, takes 7.62 (ms, per Python) / 7.55 (ms, per C)

Generating two vectors, each with 100,000.0K random doubles...
numpy:             product = 25,002,481.35212, takes 79.72 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 25,002,481.35212, takes 86.20 (ms, per Python) / 86.13 (ms, per C)
my_dot_product():  product = 25,002,481.35212, takes 86.23 (ms, per Python) / 86.16 (ms, per C)

Generating two vectors, each with 200,000.0K random doubles...
numpy:             product = 49,997,609.56441, takes 184.45 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 49,997,609.56441, takes 182.33 (ms, per Python) / 182.25 (ms, per C)
my_dot_product():  product = 49,997,609.56441, takes 161.26 (ms, per Python) / 161.18 (ms, per C)

Generating two vectors, each with 300,000.0K random doubles...
numpy:             product = 75,003,193.53317, takes 235.61 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 75,003,193.53317, takes 277.15 (ms, per Python) / 277.07 (ms, per C)
my_dot_product():  product = 75,003,193.53317, takes 220.65 (ms, per Python) / 220.58 (ms, per C)

```