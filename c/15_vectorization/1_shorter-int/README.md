# Hello World

* Vectorization with `uint_8` instead of `uint_32` can boost the performance even further--
since Intel's SIMD registers (XMM0–XMM7) are 128-bit wide, they can accommodate up to 16 `uint_8` variables but
4 `uint_32` variables only.

* Results:
```
icc, vectorization on:  Average: 16ms, std: 3.248703
icc, vectorization off: Average: 68ms, std: 8.910338
gcc, vectorization on:  Average: 24ms, std: 2.606763
gcc, vectorization off: Average: 72ms, std: 13.794717
```