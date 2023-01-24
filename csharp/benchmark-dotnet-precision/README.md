# BenchmarkDotNet

* BenchmarkDotNet is a useful tool for benchmarking. It makes writing
benchmarks "no harder than writing unit tests!"

* The issue here is: suppose we have two very fast methods (e.g. sub-μs
execution time) and try to benchmark them against each other, if we don't
include an iteration in the benchmarking functions, would the result be
inaccurate due to the accuracy of OS's timing mechanism?
    * For example, Microsoft Windows provides time resolution up to
    [100-nano seconds only](https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps),
    what if we need something much more precise than this to measure execution
    time? Can BenchmarkDotNet still cope?


## Results
``` 
BenchmarkDotNet=v0.13.4, OS=Windows 10 (10.0.19044.1526/21H2/November2021Update)
Intel Xeon CPU E5-1620 v4 3.50GHz, 1 CPU, 8 logical and 4 physical cores
.NET SDK=6.0.202
  [Host]     : .NET 6.0.7 (6.0.722.32202), X64 RyuJIT AVX2
  DefaultJob : .NET 6.0.7 (6.0.722.32202), X64 RyuJIT AVX2
```

|             Method |                 Mean |              Error |             StdDev |
|------------------- |--------------------- |------------------- |------------------- |
| IterateManyPlusOne | 12,113,032,041.67 ns | 238,654,682.282 ns | 310,318,498.471 ns |
|  IterateOnePlusOne |             12.54 ns |           0.272 ns |           0.731 ns |
|        IterateMany | 11,819,881,834.38 ns | 235,026,384.899 ns | 365,907,766.429 ns |
|         IterateOne |             11.17 ns |           0.262 ns |           0.575 ns |


* The result doesn't seem to be conclusive--guess it implies no performance
penalties?
    * But for a 3.5GHz CPU it executes 3.5 instruction cycles per ns (
i.e., 0.28 ns per cycle), how come BenchmarDotNet reports 2 decimal places for
ns?