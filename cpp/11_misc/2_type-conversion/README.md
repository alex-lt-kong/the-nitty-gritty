# Type conversion

* Type conversion includes many different functions.

## Standard conversion

* The very first type of it is called standard conversion:

    ```C
    int16_t a = 65; //0x41
    int32_t b = a;
    ```
    ```
    a: 41
    b: 41000000
    ```
    * In the 1st example, the bit representation of the variable is the same,
    we simply copy the value and pad extra memory blocks with 0's.
    * One reasonable but less obvious point is that standard conversion could
    be more than this:

        ```C
        float a = 3.14159;
        double b = a;
        double c = 3.14159;
        b == c: 0
        ```
        ```
        a: d00f4940
        b: 00000000fa210940
        c: 6e861bf0f9210940
        ```
    * If bit representations are different, it is not guaranteed that source
    value and destination value are the same.

    * There is another sneaky case, what will be the values of `b` and `c`?:
        ```C
        int32_t a = -1234567;
        uint32_t b = a;
        uint64_t c = a;
        ```
        ```
        a:   -1234567 (0x7929edff)
        b: 4293732729 (0x7929edff)
        c: 4293732729 (0x7929edffffffffff)
        ```
        * This behavior is explicitly documented in C standard 6.3.1.3:
        > 2 Otherwise, if the new type is unsigned, the value is converted by
        > repeatedly adding or subtracting one more than the maximum value
        > that can be represented in the new type until the value is in the
        > range of the new type.49)
    * The reason for this behavior is that it works best with the two's
    complement representation for negative numbers and it makes the result
    of modulo reasonable.

## Type casting

* The most common way of type-casting is the C-style one:
    ```C
    float a = 3.1415;
    int b = (int)a;
    ```
    * But this is not the one we will spend much time on... Instead, we will
    look at four C++-specific features, `dynamic_cast<new_type>()`,
    `reinterpret_cast<new_type>()`, `static_cast<new_type>()` and
    `const_cast<new_type>()`

* But wait, how exactly does the C-style work anyway?
    * In C++, a C-style cast is basically identical to trying out a range of
    sequences of C++ casts, and taking the first C++ cast that works,
    without considering dynamic_cast.
    * Using them for numeric casts should be fine and concise, but for
    C++-specific casts (i.e., casts involving object pointers/references),
    it would be safer to use the dedicated one.

### static_cast<new_type>()

* `static_cast` is used for ordinary typecasting when we know an object can be
cast to another:

    ```C++
    void func(void *data) {
        // We know data is just c, so we static_cast<> it.
        MyClass *c = static_cast<MyClass*>(data);
    }

    int main() {
        MyClass c;
        start_thread(&func, &c).join();
    }
    ```
    *No runtime checks are performed and thus users should somehow know
    they don't cast the source object into something incompatible.

### dynamic_cast<new_type>()

* The primary purpose of the dynamic_cast operator is to perform type-safe
downcasts, that is, a cast from a parent class to a child class.
    * This doesn't seem to be needed in the short term, let's skip its 
    intricacies, for now.




## References
* [cplusplus.com - Type conversions](https://cplusplus.com/doc/tutorial/typecasting/)
* [N1256 Draft of C standard](https://www.open-std.org/jtc1/sc22/WG14/www/docs/n1256.pdf)
* [stackoverflow.com - Does signed to unsigned casting in C changes the bit values](https://stackoverflow.com/questions/58415764/does-signed-to-unsigned-casting-in-c-changes-the-bit-values)
* [stackoverflow.com - Regular cast vs. static_cast vs. dynamic_cast](https://stackoverflow.com/questions/28002/regular-cast-vs-static-cast-vs-dynamic-cast)
* [IBM - The dynamic_cast operator (C++ only)](https://www.ibm.com/docs/en/zos/2.4.0?topic=expressions-dynamic-cast-operator-c-only)
