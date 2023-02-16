* In a word, per C standard, the result is undefined if the right operand is negative, or greater than or equal to
the number of bits in the left expression’s type.
* `gcc` does make some of the "undefined" behaviors "reasonable", though.
*
```
1
2
4
8
16
32
64
128
256
512
1024
2048
4096
8192
16384
32768
65536
131072
262144
524288
1048576
2097152
4194304
8388608
16777216
33554432
67108864
134217728
268435456
536870912
1073741824
2147483648
1


=====

Function result: 1
Function result: 2
Function result: 4
Function result: 8
Function result: 16
Function result: 32
Function result: 64
Function result: 128
Function result: 256
Function result: 512
Function result: 1024
Function result: 2048
Function result: 4096
Function result: 8192
Function result: 16384
Function result: 32768
Function result: 65536
Function result: 131072
Function result: 262144
Function result: 524288
Function result: 1048576
Function result: 2097152
Function result: 4194304
Function result: 8388608
Function result: 16777216
Function result: 33554432
Function result: 67108864
Function result: 134217728
Function result: 268435456
Function result: 536870912
Function result: 1073741824
Function result: 2147483648
Function result: 1
Function result: 2
Function result: 4


=====

Function result: 1
Function result: 2
Function result: 4
Function result: 8
Function result: 16
Function result: 32
Function result: 64
Function result: 128
Function result: 256
Function result: 512
Function result: 1024
Function result: 2048
Function result: 4096
Function result: 8192
Function result: 16384
Function result: 32768
Function result: 65536
Function result: 131072
Function result: 262144
Function result: 524288
Function result: 1048576
Function result: 2097152
Function result: 4194304
Function result: 8388608
Function result: 16777216
Function result: 33554432
Function result: 67108864
Function result: 134217728
Function result: 268435456
Function result: 536870912
Function result: 1073741824
Function result: 2147483648
Function result: 4294967296
Function result: 8589934592
Function result: 17179869184
```
