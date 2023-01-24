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
    no C++ functions are used at all.
    * Note that the strong form of zero-cost abstraction can never be achieved
    as even for-loop and function calls impose extra overheads (while
    compilers do try tricks such as loop unrolling and function inlining, they
    are not always performed given program size consideration, etc). As long as
    we want to keep the paradigm of "structured programming" (let alone OOP),
    some costs are inevitably involved.
    The general expectation of "zero-cost" for C++ (and Rust perhaps) is that
    they impose no extra cost compared to C.