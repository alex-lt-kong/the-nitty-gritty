# Compile-time computing

* [Compile-time function execution](https://en.wikipedia.org/wiki/Compile-time_function_execution)
or compile-time computation is a rather common and intuitive optimization
technique in C/C++.

* The idea is very simple--if a compiler can know the result of some 
computation in compile-time, it will replace the calculation with a constant,
reducing O(whatever) time complexity to O(1).

* For example, the following snippet:
    ```C
    int Factorial(int n) {
        int result = 1;
        while (n > 1) {
            result *= n--;
        }
        return result;
    }

    int main() {
        int f4 = Factorial(4);  // f4 == 24
        return f4;
    }
    ```
    can be reduced to:

    ```C
    int main() {
        return 24;
    }
    ```

* But does something similar exist in C#? We will find it out.

## Results from source code compiler

* For a very simple function:
    ```C#
        public static int GetConstantId() {
            int id = 1 + 2 + 3 + 4 + 5;
            return id;
        }
    ```
    source code compiler does turn it into a constant:
    ```C#
    .method public hidebysig static 
	int32 GetConstantId () cil managed 
    {
        // Method begins at RVA 0x20bf
        // Header size: 1
        // Code size: 3 (0x3)
        .maxstack 8

        // return 15;
        IL_0000: ldc.i4.s 15
        IL_0002: ret
    } // end of method Program::GetConstantId
    ```
    But the function is neither optimized away nor inlined:
    ```C#
    {
        // ...
        // int constantId = GetConstantId();
        IL_0000: call int32 MyProgram.Program::GetConstantId()
        IL_0005: stloc.0
        // ...
    } // end of method Program::Main

    ```

* For a slightly more complicated version:
    ```C#
    public static int Factorial(int n) {
        int result = 1;
        while (n > 1) {
            result *= n--;
        }
        return result;
    }
    ```
    almost no optimization is done at CIL bytecode level:
    ```C#
    // int value = Factorial(4);
	IL_0006: ldc.i4.4
	IL_0007: call int32 MyProgram.Program::Factorial(int32)
	IL_000c: stloc.1
    ```

* However, as C# has a second chance during the JIT compilation, this test is
not conclusive, yet.