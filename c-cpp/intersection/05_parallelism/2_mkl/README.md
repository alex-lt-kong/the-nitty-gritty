# Intel oneAPI Math Kernel Library

* Intel oneAPI Math Kernel Library (Intel MKL) is a library of optimized math routines for science,
engineering, and financial applications. The library has C and Fortran interfaces for most routines on the CPU, and DPC++ interfaces for some routines on both the CPU and GPU. 

* Intel MKL supports a wide range of calculations, for our purpose, Vector
Mathematical Functions documented [here](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/vector-mathematical-functions/vm-mathematical-functions.html) are something most likely to be useful.

## Python wrapper and benchmark against Numpy

* If the Python wrapper doesn't run and complains:
    * `INTEL MKL ERROR: ...libmkl_avx2.so.2: undefined symbol: mkl_sparse_optimize_bsr_trsm_i8`,
    may consider this per [this link](https://github.com/ikinsella/trefide/issues/2): 
    ```
    export LD_PRELOAD=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so
    ```

    * `INTEL MKL ERROR: ...libmkl_vml_avx2.so.2: undefined symbol: mkl_blas4vml_dptrmm.`,
    may consider this per [this link](https://stackoverflow.com/a/48195671/19634193): 
    ```
    export LD_PRELOAD=$LD_PRELOAD:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx.so.2:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so
    ```

* Linking options don't appear to have a major impact on performance--as long as it still compiles.
  * But Intel provides 
  [an online tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html)
  to tweak them anyway...

* According to NumPy's [document](https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries),
it usually uses an accelerated linear algebra library to improve performance--typically
either OpenBLAS or Intel MKL. To check which library it is using, issue `np.show_config()`