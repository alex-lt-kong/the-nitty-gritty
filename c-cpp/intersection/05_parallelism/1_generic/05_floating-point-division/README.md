# Floating point division

* `gdb --quiet --eval-command="disassemble /m func_floating_division" --batch ./func-no.o | tail -n +2 | head -n -1 | cut -c 4-`

* Results:
```
./main-int-on.out: avg: 42.49ms, std: 2441.59
./main-int-no.out: avg: 63.55ms, std: 1278.34
./main-flt-on.out: avg: 32.95ms, std: 1180.09
./main-flt-no.out: avg: 116.81ms, std: 1468.42
```

* Multiplication usually takes only one to two CPU clock cycles while division could take 20+ CPU clock cycles to finish
according to
[here](https://stackoverflow.com/questions/2858483/how-can-i-compare-the-performance-of-log-and-fp-division-in-c), 
[here](https://stackoverflow.com/questions/4125033/floating-point-division-vs-floating-point-multiplication) and
[here](https://www.youtube.com/watch?v=bSkpMdDe4g4). How long does it take for a CPU to read from L1, L2, L3 cache? 
According to [this post](https://www.nexthink.com/blog/smarter-cpu-testing-kaby-lake-haswell-memory/) published in 2021,
the answers are 4-5 cycles, 12 cycles and 42 cycles respectively. Therefore:
  * It is much easier to "saturate" CPU with division operations than to saturate the CPU with multiplication ones.
  * If we apply multiplication to a large array, it is likely that CPU can finish the calculation before next 
  batch of data can be fetched from memory. If this is the case, vectorization won't help much--CPU is not the
  bottleneck in the first place.
  * On the other hand, if we apply division to a large array, it is less likely that CPU can finish calculation before
  next batch of data are ready, and thus vectorized version can be observed to be much faster.
  * This thoery is consistent with what we observed from the code--vectorized multiplication is only 50% faster while
  vectorized division can be 2.5x faster.
