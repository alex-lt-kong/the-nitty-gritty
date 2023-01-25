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
    ```csharp
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
    almost no optimization is done at the CIL bytecode level:
    ```csharp
    // int value = Factorial(4);
	IL_0006: ldc.i4.4
	IL_0007: call int32 MyProgram.Program::Factorial(int32)
	IL_000c: stloc.1
    ```

* However, as C# has a second chance during the JIT compilation, this test is
not conclusive, yet.

* To dig a bit deeper, we want to examine the ultimate machine code generated
by the JIT compiler:

    ```nasm
        21:         public static void Main(string[] args)
        22:         {
        23:             int id = GetConstantId();
    00007FFF30E43FD0  push        rbp  
    00007FFF30E43FD1  sub         rsp,50h  
    00007FFF30E43FD5  lea         rbp,[rsp+50h]  
    00007FFF30E43FDA  vxorps      xmm4,xmm4,xmm4  
    00007FFF30E43FDE  vmovdqa     xmmword ptr [rbp-30h],xmm4  
    00007FFF30E43FE3  vmovdqa     xmmword ptr [rbp-20h],xmm4  
    00007FFF30E43FE8  vmovdqa     xmmword ptr [rbp-10h],xmm4  
    00007FFF30E43FED  mov         qword ptr [rbp+10h],rcx  
    00007FFF30E43FF1  call        qword ptr [CLRStub[MethodDescPrestub]@00007FFF30F1BF90 (07FFF30F1BF90h)]  
    00007FFF30E43FF7  mov         dword ptr [rbp-4],eax  
        24:             int f4 = Factorial(4);
    00007FFF30E43FFA  mov         ecx,4  
    00007FFF30E43FFF  call        qword ptr [CLRStub[MethodDescPrestub]@00007FFF30F1BF78 (07FFF30F1BF78h)]  
    00007FFF30E44005  mov         dword ptr [rbp-8],eax  
    ```
    * It seems that none of the function calls are optimized away.
    