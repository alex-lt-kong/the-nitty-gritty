# Compute a vector-vector dot product.

* It is very likely that the dot product calculation is memory-bound--given the size of the data and the predictable
pattern, prefetchers/caching should have already been fully operational. Perhaps we are
approaching the hard limit of memory bandwidth.
  * A useful [reference](https://stackoverflow.com/questions/18159455/why-vectorizing-the-loop-does-not-have-performance-improvement/18159503#18159503)

## Results
```
>>> py main.py 
Generating two vectors, each with 0.1K random doubles...
numpy:             product = 23.07125, takes 0.04 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 23.07125, takes 2.65 (ms, per Python) / 2.59 (ms, per C)
my_dot_product():  product = 23.07125, takes 0.05 (ms, per Python) / 0.00 (ms, per C)

Generating two vectors, each with 1.0K random doubles...
numpy:             product = 245.19251, takes 0.02 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 245.19251, takes 0.05 (ms, per Python) / 0.00 (ms, per C)
my_dot_product():  product = 245.19251, takes 0.05 (ms, per Python) / 0.00 (ms, per C)

Generating two vectors, each with 10.0K random doubles...
numpy:             product = 2,512.47507, takes 0.03 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 2,512.47507, takes 0.06 (ms, per Python) / 0.01 (ms, per C)
my_dot_product():  product = 2,512.47507, takes 0.06 (ms, per Python) / 0.01 (ms, per C)

Generating two vectors, each with 100.0K random doubles...
numpy:             product = 24,965.81085, takes 0.12 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 24,965.81085, takes 0.11 (ms, per Python) / 0.05 (ms, per C)
my_dot_product():  product = 24,965.81085, takes 0.29 (ms, per Python) / 0.23 (ms, per C)

Generating two vectors, each with 1,000.0K random doubles...
numpy:             product = 249,890.37412, takes 0.64 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 249,890.37412, takes 0.71 (ms, per Python) / 0.65 (ms, per C)
my_dot_product():  product = 249,890.37412, takes 0.79 (ms, per Python) / 0.72 (ms, per C)

Generating two vectors, each with 10,000.0K random doubles...
numpy:             product = 2,499,917.35075, takes 8.96 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 2,499,917.35075, takes 10.48 (ms, per Python) / 10.42 (ms, per C)
my_dot_product():  product = 2,499,917.35075, takes 8.84 (ms, per Python) / 8.77 (ms, per C)

Generating two vectors, each with 100,000.0K random doubles...
numpy:             product = 25,001,304.15563, takes 85.22 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 25,001,304.15563, takes 90.38 (ms, per Python) / 90.31 (ms, per C)
my_dot_product():  product = 25,001,304.15563, takes 83.13 (ms, per Python) / 83.06 (ms, per C)

Generating two vectors, each with 200,000.0K random doubles...
numpy:             product = 50,003,564.82433, takes 163.21 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 50,003,564.82433, takes 180.64 (ms, per Python) / 180.56 (ms, per C)
my_dot_product():  product = 50,003,564.82434, takes 160.70 (ms, per Python) / 160.62 (ms, per C)

Generating two vectors, each with 300,000.0K random doubles...
numpy:             product = 74,993,455.11691, takes 246.62 (ms, per Python) / NA    (ms, per C)
mkl_dot_product(): product = 74,993,455.11692, takes 282.95 (ms, per Python) / 282.88 (ms, per C)
my_dot_product():  product = 74,993,455.11691, takes 232.27 (ms, per Python) / 232.20 (ms, per C)

```