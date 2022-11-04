# Compute natural logarithm of vector elements

```
>>> py main.py
Generating vector vec_in with 0.1K random doubles...
numpy:    vec_out[42] = ln(0.53722) = -0.62136, takes     0.02 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[42] = ln(0.53722) = -0.62136, takes     0.13 (ms, per Python) /     0.04 (ms, per C)
mkl_ln(): vec_out[42] = ln(0.53722) = -0.62136, takes     1.74 (ms, per Python) /     1.68 (ms, per C)

Generating vector vec_in with 1.0K random doubles...
numpy:    vec_out[422] = ln(0.75843) = -0.27651, takes     0.02 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[422] = ln(0.75843) = -0.27651, takes     0.08 (ms, per Python) /     0.03 (ms, per C)
mkl_ln(): vec_out[422] = ln(0.75843) = -0.27651, takes     0.07 (ms, per Python) /     0.02 (ms, per C)

Generating vector vec_in with 10.0K random doubles...
numpy:    vec_out[878] = ln(0.04003) = -3.21816, takes     0.12 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[878] = ln(0.04003) = -3.21816, takes     0.14 (ms, per Python) /     0.10 (ms, per C)
mkl_ln(): vec_out[878] = ln(0.04003) = -3.21816, takes     0.10 (ms, per Python) /     0.04 (ms, per C)

Generating vector vec_in with 100.0K random doubles...
numpy:    vec_out[80,891] = ln(0.82843) = -0.18822, takes     0.89 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[80,891] = ln(0.82843) = -0.18822, takes     0.83 (ms, per Python) /     0.78 (ms, per C)
mkl_ln(): vec_out[80,891] = ln(0.82843) = -0.18822, takes     0.63 (ms, per Python) /     0.58 (ms, per C)

Generating vector vec_in with 1,000.0K random doubles...
numpy:    vec_out[538,771] = ln(0.43385) = -0.83505, takes    11.71 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[538,771] = ln(0.43385) = -0.83505, takes     6.99 (ms, per Python) /     6.90 (ms, per C)
mkl_ln(): vec_out[538,771] = ln(0.43385) = -0.83505, takes     3.77 (ms, per Python) /     3.71 (ms, per C)

Generating vector vec_in with 10,000.0K random doubles...
numpy:    vec_out[5,877,746] = ln(0.30579) = -1.18486, takes    74.48 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[5,877,746] = ln(0.30579) = -1.18486, takes    66.30 (ms, per Python) /    66.23 (ms, per C)
mkl_ln(): vec_out[5,877,746] = ln(0.30579) = -1.18486, takes    30.87 (ms, per Python) /    30.80 (ms, per C)

Generating vector vec_in with 100,000.0K random doubles...
numpy:    vec_out[97,090,005] = ln(0.22976) = -1.47070, takes   711.30 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[97,090,005] = ln(0.22976) = -1.47070, takes   583.16 (ms, per Python) /   583.06 (ms, per C)
mkl_ln(): vec_out[97,090,005] = ln(0.22976) = -1.47070, takes   325.89 (ms, per Python) /   325.82 (ms, per C)

Generating vector vec_in with 200,000.0K random doubles...
numpy:    vec_out[174,023,337] = ln(0.75683) = -0.27862, takes 1,451.14 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[174,023,337] = ln(0.75683) = -0.27862, takes 1,026.49 (ms, per Python) / 1,026.42 (ms, per C)
mkl_ln(): vec_out[174,023,337] = ln(0.75683) = -0.27862, takes   610.47 (ms, per Python) /   610.40 (ms, per C)

Generating vector vec_in with 400,000.0K random doubles...
numpy:    vec_out[15,423,587] = ln(0.98554) = -0.01457, takes 2,878.91 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[15,423,587] = ln(0.98554) = -0.01457, takes 2,147.34 (ms, per Python) / 2,147.27 (ms, per C)
mkl_ln(): vec_out[15,423,587] = ln(0.98554) = -0.01457, takes 1,280.01 (ms, per Python) / 1,279.93 (ms, per C)
```