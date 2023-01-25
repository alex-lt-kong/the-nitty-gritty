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

* It is unclear whether or not C# (or to be a bit more specific, Microsoft's
implementation of C# environment) follows the zero-cost abstraction approach.
Let's find it out!

## Results

```
ore Processor (Skylake, IBRS), 2 CPU, 2 logical and 2 physical cores
.NET SDK=7.0.102
  [Host]     : .NET 6.0.13 (6.0.1322.58009), X64 RyuJIT AVX2
  DefaultJob : .NET 6.0.13 (6.0.1322.58009), X64 RyuJIT AVX2
```

|                        Method |     Mean |     Error |    StdDev |
|------------------------------ |--------- |---------- |---------- |
|    StaticIterateWithInterface | 2.657 us | 0.0519 us | 0.0693 us |
|  StaticIterateWithInheritance | 2.618 us | 0.0503 us | 0.0579 us |
| StaticIterateWithoutInterface | 2.611 us | 0.0368 us | 0.0344 us |
|          IterateWithInterface | 2.610 us | 0.0490 us | 0.0482 us |
|        IterateWithInheritance | 2.647 us | 0.0465 us | 0.0777 us |
|       IterateWithoutInterface | 2.578 us | 0.0306 us | 0.0286 us |
|        IterateWithoutAnything | 2.590 us | 0.0457 us | 0.0382 us |

* The result is not very intuitive as it seems that all variants are more or
less the same.

* Examining the CIL bytecode with [ILSpy](https://github.com/icsharpcode/ILSpy)
does not reveal many useful details because CIL bytecode is mostly just a
straightforward translation of C# source code. For example, the bytecode of
`IterateWithInheritance()` and  `StaticIterateWithInheritance()`
are very similar and a just-in-time compiler has a lot of leeway to optimize
them:

    * CIL bytecode of `IterateWithInheritance()`:

    ```C#
    .method public hidebysig 
	instance void IterateWithInheritance () cil managed 
    {
        .custom instance void [BenchmarkDotNet.Annotations]BenchmarkDotNet.Attributes.BenchmarkAttribute::.ctor(int32, string) = (
            01 00 4b 00 00 00 66 43 3a 5c 55 73 65 72 73 5c
            6d 61 6d 73 64 73 5c 44 6f 63 75 6d 65 6e 74 73
            5c 72 65 70 6f 73 5c 74 68 65 2d 6e 69 74 74 79
            2d 67 72 69 74 74 79 5c 63 73 68 61 72 70 5c 61
            62 73 74 72 61 63 74 69 6f 6e 2d 6f 76 65 72 68
            65 61 64 73 5c 61 62 73 74 72 61 63 74 69 6f 6e
            2d 6f 76 65 72 68 65 61 64 73 2e 63 73 00 00
        )
        // Method begins at RVA 0x21a3
        // Header size: 1
        // Code size: 16 (0x10)
        .maxstack 8

        // new IteratorWithInheritance().Iterate(1024u);
        IL_0000: newobj instance void MyBenchmarks.IteratorWithInheritance::.ctor()
        IL_0005: ldc.i4 1024
        IL_000a: callvirt instance void MyBenchmarks.IteratorWithInheritance::Iterate(uint32)
        // }
        IL_000f: ret
    } // end of method AbstractionCostTest::IterateWithInheritance
    ```
    * CIL bytecode of `StaticIterateWithInheritance()`:

    ```C#
    .method public hidebysig 
	instance void StaticIterateWithInheritance () cil managed 
    {
        .custom instance void [BenchmarkDotNet.Annotations]BenchmarkDotNet.Attributes.BenchmarkAttribute::.ctor(int32, string) = (
            01 00 62 00 00 00 66 43 3a 5c 55 73 65 72 73 5c
            6d 61 6d 73 64 73 5c 44 6f 63 75 6d 65 6e 74 73
            5c 72 65 70 6f 73 5c 74 68 65 2d 6e 69 74 74 79
            2d 67 72 69 74 74 79 5c 63 73 68 61 72 70 5c 61
            62 73 74 72 61 63 74 69 6f 6e 2d 6f 76 65 72 68
            65 61 64 73 5c 61 62 73 74 72 61 63 74 69 6f 6e
            2d 6f 76 65 72 68 65 61 64 73 2e 63 73 00 00
        )
        // Method begins at RVA 0x21e3
        // Header size: 1
        // Code size: 16 (0x10)
        .maxstack 8

        // staticIteratorWithInheritance.Iterate(1024u);
        IL_0000: ldsfld class MyBenchmarks.IteratorWithInheritance MyBenchmarks.AbstractionCostTest::staticIteratorWithInheritance
        IL_0005: ldc.i4 1024
        IL_000a: callvirt instance void MyBenchmarks.IteratorWithInheritance::Iterate(uint32)
        // }
        IL_000f: ret
    } // end of method AbstractionCostTest::StaticIterateWithInheritance

    ```

* The above observation is consistent with
[this MSDN article](https://learn.microsoft.com/en-us/archive/msdn-magazine/2015/february/compilers-what-every-programmer-should-know-about-compiler-optimizations)
on DotNet optimization: For C# there is a source code compiler (C# compiler)
and a JIT compiler. The source code compiler performs only minor optimizations.
For example, it doesn’t perform function inlining and loop optimizations.
Instead, these optimizations are handled by the JIT compiler. The JIT compiler
that ships with the .NET Framework 4.5.1 and later versions, called RyuJIT,
supports SIMD.

