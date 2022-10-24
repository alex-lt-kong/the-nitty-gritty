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

## Results
```
>>> 
```