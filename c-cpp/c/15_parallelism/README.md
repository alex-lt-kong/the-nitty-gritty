# Parallelism

Parallelism is a huge topic... The first part of this project is on CPU-based
vectorization. 
[A Guide to Vectorization with Intel C++ Compilers](https://www.intel.com/content/dam/develop/external/us/en/documents/compilerautovectorizationguide.pdf) (the Guide)
could be a starting point for beginners. The second part of this project moves
the focus to GPU-based parallelism (Nvidia's CUDA)

## CPU-based approach (vectorization)

### Install Intel's C++ compiler

* `gcc` supports auto vectorization as well. However, since the Guide uses Intel's own compiler, here we also document
how to install and use it.

* Experiments show that neither `gcc` nor `icc` can constantly outperform the other.

* Install both the `base kit` and the `HPC kit` from: https://www.intel.com/content/www/us/en/developer/articles/news/free-intel-software-developer-tools.html

* Follow the steps to configure the environment: https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-hpc-linux/top.html

* `~/intel/oneapi/setvars.sh` does not seem to work properly on my machine, the absolute path of the compiler should be at
`/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icc`. Note that the same compiler is called `icl` on Windows
and `icc` on Linux.

* The `-vec-report` option documented in the Guide seems obsolete as of 2022, use `-qopt-report -qopt-report-phase=vec`
instead.

### Points to note:

* The performance gain from vectorization can be difficult to measure, since CPU caching, memory bandwidth, etc 
 can all have an even larger impact on the performance.

* Check CPU cache: `lscpu | grep cache` (Note that L1 cache is usually reported on a per-core basis. However, L3 is
mostly shared by all cores. How about L2? It dpends lol)
```
L1d cache:                       64 KiB
L1i cache:                       64 KiB
L2 cache:                        512 KiB
L3 cache:                        10 MiB
```
* Let's say, fetching an integer from memory takes one second and `add` operation takes one nanosecond. What
will be the CPU's utilization during the execution of `add eax,DWORD PTR [rax]`?
  * My answer: the CPU will be 100% utilized for 1 second + 1 nanosecond, instead of 0% utilized for
  1 second + 100% utilized for 1 nano second.
  * One implication is, if CPU is now 100% utilized, there is no easy way for us to know if CPU is "really"
  doing calculation or it is "idly" waiting for data to arrive from memory.
  * This is not the case for reading data from hard drives, though. Suppose it takes one second to read an integer
  from hard drives to memory, one nanosecond to read the integer from memory to register and one nanosecond to
  execution the `add` operation. The CPU should be 0% utilized for 1 second and 100% utilized for just 2 nanoseconds.

* To make the performance gain more pronunced in experiments, this project takes the following two approaches:
  * Limit the size of test array to the size of L1 cache (i.e., not greater than a few hundred KBs), so that we eliminate
  the bottleneck at memory bandwidth.
  * use more time-consuming (but still vectorized) CPU instructions such as `divps` (i.e., performs a SIMD divide
  of the 4/8/16 packed single-precision floating-point values in the first source operand by the 4/8/16 packed
  single-precision floating-point values in the second source operand) instead of simpler SIMD instructions such
  as `addps` (Add 4/8/16 packed single-precision floating-point values from the first source operand with the second
  source operand, and stores the packed single-precision floating-point results in the destination operand.)

* Some experiments which are more closely related to memory are moved to the separate directory
[01_memory-and-cpu-cache](./01_memory-and-cpu-cache).

## GPU-based approach