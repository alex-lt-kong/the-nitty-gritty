# Abstraction costs

* One of the founding principles of C++ (and Rust as well) is the [zero-overhead
principle](https://en.cppreference.com/w/cpp/language/Zero-overhead_principle).
Bjarne Stroustrup states that it means:
    * You don't pay for what you don't use.
    * What you do use is just as efficient as what you could reasonably write
    by hand.

* Unfortunately, the principle of zero overheads isn't really the topic of
this test. This test focuses on the idea of "zero-overhead abstraction"
(a.k.a. "zero-cost abstraction"), which generally means "I don't pay for what
I *do* use" as discussed in this [CppCon seminar](https://isocpp.org/blog/2020/07/cppcon-2019-there-are-no-zero-cost-abstractions-chandler-carruth)
and [this forum](https://news.ycombinator.com/item?id=20948118).
    * The idea of zero-cost abstraction is closely related to C++'s "as-if" rule
    explained in the templates feature [here](../../cpp/04_templates/). It 
    roughly means that hopefully all abstractions provided by C++ are optimized
    away at the machine code level so that machine code should be as fast as
    no C++ abstractions are used at all.
        * Theoretically speaking, one can also argue that there are at least
        three types of costs: runtime cost, build time cost and human cost.
        For this project, we are mostly focusing on runtime cost (rarely, on
        build time cost perhaps) but never on human costs.
    * Note that the strong form of zero-cost abstraction can never be achieved
    as even for-loop and function calls impose extra overheads (while
    compilers do try tricks such as loop unrolling and function inlining, they
    are not always performed given program size consideration, etc). As long as
    we want to keep the paradigm of "structured programming" (let alone OOP),
    some costs are inevitably involved.
    * The general expectation of "zero-cost" for C++ (and Rust perhaps) is that
    they impose no extra cost compared to C.

IterateWithInterface: 2153 ns
IterateWithInheritance: 2167 ns
IterateWithoutInterface: 2173 ns
IterateWithoutAnything: 2115 ns

|                  Method |     Mean |     Error |    StdDev |
|------------------------ |--------- |---------- |---------- |
|    IterateWithInterface | 2.139 us | 0.0308 us | 0.0273 us |
|  IterateWithInheritance | 2.129 us | 0.0409 us | 0.0455 us |
| IterateWithoutInterface | 2.104 us | 0.0410 us | 0.0384 us |
|  IterateWithoutAnything | 2.131 us | 0.0426 us | 0.0491 us |

```
BenchmarkDotNet=v0.13.4, OS=Windows 10 (10.0.19044.1526/21H2/November2021Update)
Intel Xeon CPU E5-1620 v4 3.50GHz, 1 CPU, 8 logical and 4 physical cores
.NET SDK=6.0.202
  [Host]     : .NET 6.0.7 (6.0.722.32202), X64 RyuJIT AVX2
  DefaultJob : .NET 6.0.7 (6.0.722.32202), X64 RyuJIT AVX2

```