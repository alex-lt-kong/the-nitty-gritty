* The very first attempt to show how vectorization can make a difference

* Results:
```
icc, vectorization on:  Average: 66ms, std: 16.943430
icc, vectorization off: Average: 111ms, std: 19.272556
gcc, vectorization on:  Average: 102ms, std: 13.690416
gcc, vectorization off: Average: 114ms, std: 20.553469
```