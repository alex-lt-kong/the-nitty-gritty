# Memory bandwitdh benchmarking

* It takes time to move data from memory to CPU and here we try to measure that.
(fun fact: if the CPU is 10 centimeters away from memory, the lower limit of
time needed is around 0.33 nanoseconds, or 0.00033 microseconds, which 
is imposed by the speed of light)


 * `lscpu | grep cache` (L1d is for data and L1i is for instruction):
```
L1d cache:                       128 KiB
L1i cache:                       128 KiB
L2 cache:                        1 MiB
L3 cache:                        8 MiB
```

* `lshw -short -C memory`
```
H/W path             Device     Class          Description
==========================================================
/0/2                            memory         16GiB System Memory
/0/2/0                          memory         8GiB Row of chips LPDDR3 Synchronous 2133 MHz (0.5 ns)
/0/2/1                          memory         [empty]
/0/2/2                          memory         8GiB Row of chips LPDDR3 Synchronous 2133 MHz (0.5 ns)
/0/2/3                          memory         [empty]
/0/e                            memory         256KiB L1 cache
/0/f                            memory         1MiB L2 cache
/0/10                           memory         8MiB L3 cache
/0/13                           memory         128KiB BIOS
/0/100/14.2                     memory         RAM memory
```

* Results (from a physical machine):
```
TESTMEM = 10240 MB
Size (kB)	Bandwidth (GB/s)	Iterations	Addresses
      8		102.64      			1310720		26e502a0
     16		122.27						655360		26e502a0
     24		124.00						436906		26e502a0
     28		121.47						374491		26e502a0
     32		119.20						327680		26e502a0
     36		64.48 						291271		26e502a0
     40		64.52 						262144		26e502a0
     48		65.05 						218453		26e502a0
     64		62.90 						163840		26e502a0
    128		60.71 						81920 		26e502a0
    256		51.49 						40960 		6c258010
    384		51.46 						27306 		6c238010
    512		51.27 						20480 		6c218010
    768		50.86 						13653 		6c1d8010
   1024		50.54 						10240 		6c198010
   1025		55.22 						10230 		26e502a0
   2048		47.75 						5120  		6c098010
   4096		49.48 						2560  		6be98010
   8192		27.65 						1280  		6ba98010
  16384		21.75 						640	  	  6b298010
 200000		18.21 						52	  	  5ff48010
 ```

* As answers in [this post](https://stackoverflow.com/questions/30313600/why-does-my-8m-l3-cache-not-provide-any-benefit-for-arrays-larger-than-1m?noredirect=1&lq=1) say, it is not easy to give conclusive explanation on why
the performace drops in some cases...