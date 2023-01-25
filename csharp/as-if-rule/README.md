# The as-if rule

* In C/C++, the ["as-if" rule](https://en.cppreference.com/w/cpp/language/as_if)
means that "all transformations that do not affect a program's 'observable
behavior' are allowed."
    * The exact definition of "observable behavior" is a bit more complicated,
    but basically, it means "the result of the program is correct".

* For example, if we compile the following C program:
    ```C
    #include <stdlib.h>

    int Factorial(int n) {
        int result = 1;
        while (n > 1) {
            result *= n--;
        }
        return result;
    }

    int main() {
        int f4 = Factorial(4);  // f4 == 24
        return 0;
    }
    ```
    Will the compiler faithfully calculate `4!` and then throw the result away?
    No, as the compiler knows the variable from the function call is never used,
    it will wisely just skip it:
    ```asm
    0000000000001040 <main>:
        1040:	31 c0                	xor    eax,eax
        1042:	c3                   	ret    
        1043:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
        104a:	00 00 00 
        104d:	0f 1f 00             	nop    DWORD PTR [rax]
    ```
    * As the program always returns zero whatsoever, as long as it returns
    0, compilers are allowed to do anything.
    * This example is almost the same as
    [the example](../compile-time-computing/c/main.c)
    in the [Compile-time computing project](../compile-time-computing/), but
    they aren't identical.

* It seems that C#'s standard does not really mention something similar. So
will it somehow follow the same approach?

* No, seems that C#'s CIL bytecode will keep the function call anyway:
    ```C#
    .method public hidebysig static 
        void Main (
            string[] args
        ) cil managed 
    {
        .custom instance void System.Runtime.CompilerServices.NullableContextAttribute::.ctor(uint8) = (
            01 00 01 00 00
        )
        // Method begins at RVA 0x20bf
        // Header size: 1
        // Code size: 8 (0x8)
        .maxstack 8
        .entrypoint

        // Factorial(4);
        IL_0000: ldc.i4.4
        IL_0001: call int32 MyProgram.Program::Factorial(int32)
        IL_0006: pop
        // }
        IL_0007: ret
    } // end of method Program::Main
    ```

* The same is seen on the machine code level:
    ```asm
        16:         public static void Main(string[] args)
        17:         {
        18:             int f4 = Factorial(4);
    00007FFF30E63FD0  push        rbp  
    00007FFF30E63FD1  sub         rsp,20h  
    00007FFF30E63FD5  lea         rbp,[rsp+20h]  
    00007FFF30E63FDA  mov         qword ptr [rbp+10h],rcx  
    00007FFF30E63FDE  mov         ecx,4  
    00007FFF30E63FE3  call        qword ptr [CLRStub[MethodDescPrestub]@00007FFF30F3BF78 (07FFF30F3BF78h)]  
        19:             Debugger.Break();
    00007FFF30E63FE9  call        qword ptr [CLRStub[MethodDescPrestub]@00007FFF31073030 (07FFF31073030h)]
    ```
