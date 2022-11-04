# Computes a to the power b for elements of two vectors. 

```
>>> py main.py
Generating vector vec_in with 0.1K random doubles...
numpy:     vec_out[12] = 0.26187^0.95980 = 0.27637, takes     0.03 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[12] = 0.26187^0.95980 = 0.27637, takes     1.85 (ms, per Python) /     1.77 (ms, per C)
my_ln():   vec_out[12] = 0.26187^0.95980 = 0.27637, takes     0.10 (ms, per Python) /     0.04 (ms, per C)

Generating vector vec_in with 1.0K random doubles...
numpy:     vec_out[893] = 0.05067^0.07973 = 0.78838, takes     0.04 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[893] = 0.05067^0.07973 = 0.78838, takes     0.11 (ms, per Python) /     0.05 (ms, per C)
my_ln():   vec_out[893] = 0.05067^0.07973 = 0.78838, takes     0.08 (ms, per Python) /     0.03 (ms, per C)

Generating vector vec_in with 10.0K random doubles...
numpy:     vec_out[1,820] = 0.37442^0.20950 = 0.81399, takes     0.24 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[1,820] = 0.37442^0.20950 = 0.81399, takes     0.17 (ms, per Python) /     0.12 (ms, per C)
my_ln():   vec_out[1,820] = 0.37442^0.20950 = 0.81399, takes     0.16 (ms, per Python) /     0.10 (ms, per C)

Generating vector vec_in with 100.0K random doubles...
numpy:     vec_out[75,732] = 0.81968^0.74162 = 0.86289, takes     2.45 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[75,732] = 0.81968^0.74162 = 0.86289, takes     0.89 (ms, per Python) /     0.83 (ms, per C)
my_ln():   vec_out[75,732] = 0.81968^0.74162 = 0.86289, takes     1.24 (ms, per Python) /     1.18 (ms, per C)

Generating vector vec_in with 1,000.0K random doubles...
numpy:     vec_out[667,689] = 0.17728^0.38416 = 0.51448, takes    23.63 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[667,689] = 0.17728^0.38416 = 0.51448, takes     8.61 (ms, per Python) /     8.54 (ms, per C)
my_ln():   vec_out[667,689] = 0.17728^0.38416 = 0.51448, takes     9.90 (ms, per Python) /     9.84 (ms, per C)

Generating vector vec_in with 10,000.0K random doubles...
numpy:     vec_out[6,231,475] = 0.33281^0.37012 = 0.66551, takes   213.98 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[6,231,475] = 0.33281^0.37012 = 0.66551, takes    74.61 (ms, per Python) /    74.54 (ms, per C)
my_ln():   vec_out[6,231,475] = 0.33281^0.37012 = 0.66551, takes    93.11 (ms, per Python) /    93.04 (ms, per C)

Generating vector vec_in with 100,000.0K random doubles...
numpy:     vec_out[75,718,606] = 0.46939^0.02325 = 0.98257, takes 2,107.93 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[75,718,606] = 0.46939^0.02325 = 0.98257, takes   745.25 (ms, per Python) /   745.18 (ms, per C)
my_ln():   vec_out[75,718,606] = 0.46939^0.02325 = 0.98257, takes   955.98 (ms, per Python) /   955.90 (ms, per C)

Generating vector vec_in with 200,000.0K random doubles...
numpy:     vec_out[154,537,479] = 0.91119^0.64248 = 0.94200, takes 4,495.16 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[154,537,479] = 0.91119^0.64248 = 0.94200, takes 1,515.62 (ms, per Python) / 1,515.55 (ms, per C)
my_ln():   vec_out[154,537,479] = 0.91119^0.64248 = 0.94200, takes 1,916.75 (ms, per Python) / 1,916.68 (ms, per C)

Generating vector vec_in with 400,000.0K random doubles...
numpy:     vec_out[105,634,148] = 0.79236^0.63070 = 0.86347, takes 8,645.34 (ms, per Python) /       NA (ms, per C)
mkl_pow(): vec_out[105,634,148] = 0.79236^0.63070 = 0.86347, takes 3,014.38 (ms, per Python) / 3,014.29 (ms, per C)
my_ln():   vec_out[105,634,148] = 0.79236^0.63070 = 0.86347, takes 3,930.80 (ms, per Python) / 3,930.72 (ms, per C)
```