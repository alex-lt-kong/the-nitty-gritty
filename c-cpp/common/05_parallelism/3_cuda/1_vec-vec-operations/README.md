# Vector addition

* No, it is not guaranteed that using GPU improves performance as it takes
quite a while to move data between main memory and GPU memory.

```
CPU: Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz
GPU: NVIDIA GeForce RTX 3060

3051.8 MB random data generated

========== Now running: vectorAdd ==========
--- Running on CPU ---
Done, took 748.33ms
--- Running on GPU ---
Took 360.71ms to move data from RAM to GPU memory (8460.4MB/sec)
Took 0.46ms to calculate
Took 908.80ms to move data from GPU memory to RAM (1679.0MB/sec)
Done, took 1392.87ms

Checking if CPU/GPU results are identical...YES!


========== Now running: vectorMul ==========
--- Running on CPU ---
Done, took 244.22ms
--- Running on GPU ---
Took 349.60ms to move data from RAM to GPU memory (8729.3MB/sec)
Took 0.04ms to calculate
Took 218.15ms to move data from GPU memory to RAM (6994.7MB/sec)
Done, took 582.80ms

Checking if CPU/GPU results are identical...YES!


========== Now running: vectorDiv ==========
--- Running on CPU ---
Done, took 257.26ms
--- Running on GPU ---
Took 365.93ms to move data from RAM to GPU memory (8339.8MB/sec)
Took 0.04ms to calculate
Took 228.29ms to move data from GPU memory to RAM (6683.9MB/sec)
Done, took 618.66ms

Checking if CPU/GPU results are identical...YES!


========== Now running: vectorPow ==========
--- Running on CPU ---
Done, took 3643.34ms
--- Running on GPU ---
Took 338.75ms to move data from RAM to GPU memory (9008.9MB/sec)
Took 0.06ms to calculate
Took 395.69ms to move data from GPU memory to RAM (3856.2MB/sec)
Done, took 769.97ms

Checking if CPU/GPU results are identical...YES!

```