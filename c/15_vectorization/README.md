# Vectorization

Vectorization is a huge topic... 
[A Guide to Vectorization with Intel C++ Compilers](https://www.intel.com/content/dam/develop/external/us/en/documents/compilerautovectorizationguide.pdf) (the Guide)
could be a starting point for beginners.

## Install Intel's C++ compiler

* `gcc` supports auto vectorization as well. However, since the Guide uses Intel's own compiler, here we also document
how to install and use it.

* Also, experiments appear to show that `gcc` is not as good as `icc` in terms of vectorization.

* Install both the `base kit` and the `HPC kit` from: https://www.intel.com/content/www/us/en/developer/articles/news/free-intel-software-developer-tools.html

* Follow the steps to configure the environment: https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-hpc-linux/top.html

* `~/intel/oneapi/setvars.sh` does not seem to work properly on my machine, the absolute path of the compiler should be at
`/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icc`. Note that the same compiler is called `icl` on Windows
and `icc` on Linux.

* The `-vec-report` option documented in the Guide seems obsolete as of 2022, use `-qopt-report -qopt-report-phase=vec`
instead.

## Points to note:

* The performance gain from vectorization can be difficult to measure, since CPU caching, memory bandwidth, etc 
 can all have an even larger impact on the performance.

* Check CPU cache: `lscpu | grep cache`
```
L1d cache:                       64 KiB
L1i cache:                       64 KiB
L2 cache:                        512 KiB
L3 cache:                        10 MiB
```
