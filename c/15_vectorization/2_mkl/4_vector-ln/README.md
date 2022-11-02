# Intel oneMKL

```
>>> py main.py 
Generating vector vec_in with 0.1K random doubles...
numpy:    vec_out[23] = -1.52022, takes     0.02 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[23] = -1.52022, takes     0.12 (ms, per Python) /     0.03 (ms, per C)
mkl_ln(): vec_out[23] = -1.52022, takes     2.06 (ms, per Python) /     1.97 (ms, per C)

Generating vector vec_in with 1.0K random doubles...
numpy:    vec_out[55] = -0.15422, takes     0.02 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[55] = -0.15422, takes     0.08 (ms, per Python) /     0.03 (ms, per C)
mkl_ln(): vec_out[55] = -0.15422, takes     0.09 (ms, per Python) /     0.02 (ms, per C)

Generating vector vec_in with 10.0K random doubles...
numpy:    vec_out[4,524] = -0.21361, takes     0.12 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[4,524] = -0.21361, takes     0.13 (ms, per Python) /     0.08 (ms, per C)
mkl_ln(): vec_out[4,524] = -0.21361, takes     0.09 (ms, per Python) /     0.04 (ms, per C)

Generating vector vec_in with 100.0K random doubles...
numpy:    vec_out[15,055] = -0.27426, takes     0.84 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[15,055] = -0.27426, takes     0.86 (ms, per Python) /     0.81 (ms, per C)
mkl_ln(): vec_out[15,055] = -0.27426, takes     0.26 (ms, per Python) /     0.21 (ms, per C)

Generating vector vec_in with 1,000.0K random doubles...
numpy:    vec_out[771,189] = -0.95029, takes     8.04 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[771,189] = -0.95029, takes     6.97 (ms, per Python) /     6.90 (ms, per C)
mkl_ln(): vec_out[771,189] = -0.95029, takes     2.16 (ms, per Python) /     2.10 (ms, per C)

Generating vector vec_in with 10,000.0K random doubles...
numpy:    vec_out[9,445,108] = -2.08092, takes    74.75 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[9,445,108] = -2.08092, takes    83.84 (ms, per Python) /    83.77 (ms, per C)
mkl_ln(): vec_out[9,445,108] = -2.08092, takes    30.87 (ms, per Python) /    30.81 (ms, per C)

Generating vector vec_in with 100,000.0K random doubles...
numpy:    vec_out[5,785,145] = -1.49069, takes   773.09 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[5,785,145] = -1.49069, takes   744.60 (ms, per Python) /   744.53 (ms, per C)
mkl_ln(): vec_out[5,785,145] = -1.49069, takes   323.40 (ms, per Python) /   323.29 (ms, per C)

Generating vector vec_in with 200,000.0K random doubles...
numpy:    vec_out[73,654,652] = -0.60519, takes 1,602.05 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[73,654,652] = -0.60519, takes 1,453.87 (ms, per Python) / 1,453.80 (ms, per C)
mkl_ln(): vec_out[73,654,652] = -0.60519, takes   657.51 (ms, per Python) /   657.43 (ms, per C)

Generating vector vec_in with 400,000.0K random doubles...
numpy:    vec_out[97,602,156] = -0.00455, takes 3,228.52 (ms, per Python) /       NA (ms, per C)
my_ln():  vec_out[97,602,156] = -0.00455, takes 2,959.35 (ms, per Python) / 2,959.26 (ms, per C)
mkl_ln(): vec_out[97,602,156] = -0.00455, takes 1,321.95 (ms, per Python) / 1,321.87 (ms, per C)
```