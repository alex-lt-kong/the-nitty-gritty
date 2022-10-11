# Data dependency

* This issue has various forms, the most obvious ones include:
  * Read-after-write: `for (j=1; j<MAX; j++) A[j]=A[j-1]+1;`.
  * Write-after-read: `for (j=1; j<MAX; j++) A[j-1]=A[j]+1;`.

* 

## Results

* `gcc`:
```
stride: 1,      0.000153303,          18
stride: 4,      0.000024557,           0
```
* `icc`:
```
stride: 1,      0.000073195,         166
stride: 4,      0.000010729,           0
```

## Caveat

* Compilers may not be able to vectorize the below loop, even if it does not appear to have any data dependency.
```
for (i = 0; i < size; i++) {
  c[i] = a[i] * b[i];
}
```

* The issue is that, if we pass `a`, `b` and `c` as three pointers, instead of `malloc()`ing memory to them in-place,
there is no way for a compiler to be sure if there are some overlaps among these memory blocks.

* If we are sure there will never be data denepdency issues, we may hint compilers like the follows:
```
if defined( __INTEL_COMPILER)
#pragma ivdep
// Pragmas are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.
// https://stackoverflow.com/questions/5078679/what-is-the-scope-of-a-pragma-directive
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
```