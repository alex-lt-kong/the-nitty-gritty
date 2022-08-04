# Vectorization in object files

* Results for `float`:
```
icc, vectorization on:  1427.53ms
icc, vectorization off: 3567.30ms (2.50x)
gcc, vectorization on:  1629.70ms
gcc, vectorization off: 4654.37ms (2.86x)
```

* Results for `double`:
```
icc, vectorization on:  2080.76ms
icc, vectorization off: 3552.29ms (1.71x)
gcc, vectorization on:  2602.55ms
gcc, vectorization off: 4687.85ms (1.80x)
```

* The non-vectorized `double`/`float` versions are very close to each other (`3567.30ms` vs `3552.29ms` for `icc`
and `4654.37ms` vs `4687.85ms` for `gcc`)

* Difference between vectorized versions are much more pronounced (`1427.53ms` vs `2080.76ms` for `icc` and
`1629.70ms` vs `1629.70ms` for `gcc`)

* `float` is 32-bit long and `double` is 64-bit long. SIMD registers (XMM0–XMM7) are 128-bit long, thus they are able to
hold up to four `float`s and two `double`s. This also indicates the upper limit of vectorization--for `float`,
the maximum performace gain is 4x and for `double` it is 2x.