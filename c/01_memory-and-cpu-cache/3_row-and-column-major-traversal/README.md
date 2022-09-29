# Row-major and column-major traversal


## Introductions

* Prefering row-major rather than column-major traversal is a relatively common optimization
technique (well, at least on C and the sort of "C-series" languages like C++/Python/JavaScript/etc).
However, it is not really easy to properly design an experiment that demonstrate this:
* The most straightforward way to define a 2D array, `int arr[d][d]`, uses stack memory, which 
imposes significant restrictions on array size (no more than a few KB in many cases).
  * If the scale of the issue is limited to a few KB, row-major and column-major won't make a signficant difference
  anyway
* We may define the array by `int* arr = malloc(d * d * sizeof(int))` and access its element by `*(arr + i * d + j)`,
but it obfuscates our purpose of testing a two-dimensional array. Compilers may not be able to recognize the pattern as
2D array access, which could break the optimization.
* We may also define the 2D array by defining an array of pointers:
  ```
  int** arr = (int**)malloc(d * sizeof(int*));
  for (i = 0; i < d; i++)
    arr[i] = (int*)malloc(i * sizeof(int));
  ```
  However, it means that we only guarantee that each sub-array is contiguous and the entire 2d array is most likely
  separate, which is not really the same as `int arr[d][d]`
* In this project, we will take the second approach, `int* arr = malloc(d * d * sizeof(int))`

## Results

* 1st.c:

```
Dim,	ArraySize(KB),	Row-Major Time,	RM Sample,	Col-Major Time,	CM Sample
   10,	          0,	0.000000000,	      15,	0.000000238,	      32
   20,	          1,	0.000000238,	      21,	0.000000477,	      26
   50,	          9,	0.000002623,	      73,	0.000001669,	     144
  100,	         39,	0.000014305,	      68,	0.000005722,	     188
  150,	         87,	0.000024557,	     186,	0.000012159,	     442
  200,	        156,	0.000079155,	     190,	0.000035286,	     186
  500,	        976,	0.000470161,	     568,	0.000353813,	     418
 1000,	       3906,	0.001616716,	    1103,	0.001428366,	     681
 2000,	      15625,	0.005551815,	    1419,	0.007848501,	    1189
 5000,	      97656,	0.035605431,	    6204,	0.060495377,	    3430
10000,	     390625,	0.148145676,	    6290,	0.463856697,	   10920
20000,	    1562500,	0.700424433,	   10111,	4.836644173,	    8593
40000,	    6250000,	6.095543385,	    9024,	44.460029602,	   18816
```

<img src="./1st.png">

// https://stackoverflow.com/questions/73891330/why-column-major-traversal-is-actually-faster-than-row-major-traversal-when-2d-a?noredirect=1#comment130472635_73891330