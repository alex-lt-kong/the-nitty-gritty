# Compute natural logarithm of vector elements

```
>>> py main.py
Generating vector vec_in with 0.1K random doubles...
numpy:    vec_out[24] = -0.08858, takes     0.02 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[24] = -0.08858, takes     0.12 (ms, per Python) /     0.04 (ms, per C)
mkl_ln(): vec_out[24] = -0.08858, takes     2.95 (ms, per Python) /     2.86 (ms, per C)

Generating vector vec_in with 1.0K random doubles...
numpy:    vec_out[869] = -1.07747, takes     0.02 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[869] = -1.07747, takes     0.09 (ms, per Python) /     0.02 (ms, per C)
mkl_ln(): vec_out[869] = -1.07747, takes     0.09 (ms, per Python) /     0.03 (ms, per C)

Generating vector vec_in with 10.0K random doubles...
numpy:    vec_out[6,360] = -0.43418, takes     0.11 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[6,360] = -0.43418, takes     0.11 (ms, per Python) /     0.06 (ms, per C)
mkl_ln(): vec_out[6,360] = -0.43418, takes     0.09 (ms, per Python) /     0.04 (ms, per C)

Generating vector vec_in with 100.0K random doubles...
numpy:    vec_out[55,860] = -0.20601, takes     1.47 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[55,860] = -0.20601, takes     0.68 (ms, per Python) /     0.63 (ms, per C)
mkl_ln(): vec_out[55,860] = -0.20601, takes     0.42 (ms, per Python) /     0.37 (ms, per C)

Generating vector vec_in with 1,000.0K random doubles...
numpy:    vec_out[739,937] = -0.31635, takes     7.50 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[739,937] = -0.31635, takes     5.48 (ms, per Python) /     5.42 (ms, per C)
mkl_ln(): vec_out[739,937] = -0.31635, takes     2.05 (ms, per Python) /     2.00 (ms, per C)

Generating vector vec_in with 10,000.0K random doubles...
numpy:    vec_out[9,412,973] = -0.48367, takes    90.58 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[9,412,973] = -0.48367, takes    49.98 (ms, per Python) /    49.90 (ms, per C)
mkl_ln(): vec_out[9,412,973] = -0.48367, takes    34.39 (ms, per Python) /    34.32 (ms, per C)

Generating vector vec_in with 100,000.0K random doubles...
numpy:    vec_out[91,142,808] = -0.96929, takes   839.02 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[91,142,808] = -0.96929, takes   650.26 (ms, per Python) /   650.19 (ms, per C)
mkl_ln(): vec_out[91,142,808] = -0.96929, takes   336.00 (ms, per Python) /   335.91 (ms, per C)

Generating vector vec_in with 200,000.0K random doubles...
numpy:    vec_out[84,000,694] = -0.45215, takes 1,902.82 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[84,000,694] = -0.45215, takes 1,216.96 (ms, per Python) / 1,216.87 (ms, per C)
mkl_ln(): vec_out[84,000,694] = -0.45215, takes   681.85 (ms, per Python) /   681.75 (ms, per C)

Generating vector vec_in with 400,000.0K random doubles...
numpy:    vec_out[168,420,619] = -0.54444, takes 3,581.10 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[168,420,619] = -0.54444, takes 2,322.10 (ms, per Python) / 2,322.00 (ms, per C)
mkl_ln(): vec_out[168,420,619] = -0.54444, takes 1,319.14 (ms, per Python) / 1,319.07 (ms, per C)
```