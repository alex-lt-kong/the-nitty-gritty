# Hello World

* Vectorization with `uint_8` instead of `uint_32` can boost the performance even further--Intel's SIMD registers
(XMM0–XMM7) are 128-bit wide, they can accommodate up to 16 `uint_8` variables but 4 `uint_32` variables only.

* Results:
```
icc, vectorization on:   618.45ms
icc, vectorization off: 4381.49ms
gcc, vectorization on:   608.03ms
gcc, vectorization off: 4706.47ms
```
