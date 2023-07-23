# Vector addition

* No, it is not guaranteed that using GPU improves performance as it takes
quite a while to move data between main memory and GPU memory.

```
=== Running on CPU ===
Done, took 1722.02ms

=== Running on GPU (NVIDIA GeForce RTX 3060) ===
Took 824.82ms to move data from RAM to GPU memory (9249.7MB/sec)
Took 1.98ms to calculate
Took 3155.73ms to move data from GPU memory to RAM (1208.8MB/sec)
Done, took 4220.42ms

Checking if CPU/GPU results are identical...
YES!
```