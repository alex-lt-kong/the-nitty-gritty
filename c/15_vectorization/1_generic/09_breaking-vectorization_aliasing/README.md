# Aliasing

* In computing, aliasing describes a situation in which a data location in
memory can be accessed through different symbolic names in the program. 

    * In C, this is typically achieved by the use of aliased pointers.
    * Interestingly, aliasing a pointer is usually prohibited in Fortran and
    thus a Fortran compiler can optmize more aggressively than a comparable
    C compiler, which, [argued by some people](https://stackoverflow.com/questions/146159/is-fortran-easier-to-optimize-than-c-for-heavy-calculations),
    is also one important reason why Fortran tends to
    be even faster then C in numeric calculations.

* Compilers may not be able to vectorize the below loop, even if it does not appear to have any data dependency.
    ```
    for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
    }
    ```

    * The issue is that, if we pass `a`, `b` and `c` as three pointers,
    instead of `malloc()`ing memory to them in-place, there is no way for a
    compiler to be sure if there are some overlaps among these memory blocks.

## Helping compilers to vectorize

* There are at least two ways we can tell a compiler that aliasing doesn't happen--so that it can vectorize 
confidently.

* The `ivdep` directive (a.k.a. pragma).
    * Its exact meaning is compiler-defined. Usually it is used to instruct
    the compiler to ignore assumed vector dependencies
    * A `gcc`/`icc` compatible version of the loop would be:
    ```
    #if defined( __INTEL_COMPILER)
    #pragma ivdep
    #elif defined(__GNUC__)
    #pragma GCC ivdep
    #endif
    for (int i = 0; i < arr_len; ++i) {
        results->r[i] = arr->r[i] / a;
        results->g[i] = b / arr->g[i];
        results->b[i] = arr->b[i] / c;
    }
    ```

* The `restrict` keyword.
    * The idea is roughly the same--by adding this type qualifier, a programmer hints to the compiler that
    for the lifetime of the pointer, no other pointer will be used to access the object to which it points.
    * It is defined in the C99 standard. If the declaration of intent is not followed and the object
    is accessed by an independent pointer, this will result in undefined behavior.
    * To use the keyword, first enable the `-fopt-info-vec-missed` option (in the case of `gcc`) to examine which
    pointer is breaking vectorization and then we add the `restrict` accordingly.
    ```
    struct restrictPixelArray {
        float* restrict r;
        float* restrict g;
        float* restrict b;
    };
    ```



* Note that it is programers' responsibility to ensure that aliasing doesn't happen--all these methods are only used
to tell compilers to vectorize even if it has concerns over aliasing and they don't really remove
aliasing if it does exist.