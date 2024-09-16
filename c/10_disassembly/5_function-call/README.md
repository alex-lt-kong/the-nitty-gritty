# Function call

* This example is mostly used to demonstrate the calling convention, which
is also explained [here](../README.md)

```asm
0000000000001135 <add>:
    1135:	push   rbp                       # previous base pointer's address is pushed to the call stack...
    1136:	mov    rbp,rsp                   # so that we can assign new stack pointer and the new base pointer
                                             # Note that we don't have sub rsp, 0xsomething here.
    1139:	mov    QWORD PTR [rbp-0x18],rdi  # a/product
    113d:	mov    QWORD PTR [rbp-0x20],rsi  # b/multiplicand
    1141:	mov    QWORD PTR [rbp-0x8],0x0   # sum
    1149:	mov    rdx,QWORD PTR [rbp-0x18]
    114d:	mov    rax,QWORD PTR [rbp-0x20]
    1151:	add    rax,rdx
    1154:	mov    QWORD PTR [rbp-0x8],rax   # stores a + b to *(rbp-0x8)
    1158:	mov    rax,QWORD PTR [rbp-0x8]   # rax stores return value   
                                             # immediately before pop, rbp stores the memory address of the stack frame base of add()
    115c:	pop    rbp                       # Now we overwrite rbp with the top most value in stack, which is the memory address of the stack frame base of multiply(). Note that rsp is changed only once during the add() function,
    which is at the very beginning of the function call at 1135
    # As there is no enter (no sub rsp, 0xsomething), there is no leave
    115d:	ret                              # pop's another element from the call stack, i.e., the return address, which was push'ed to the call stack at 1118e

000000000000115e <multiply>:
    115e:	push   rbp                        # previous base pointer's address is pushed to the call stack...
    115f:	mov    rbp,rsp                    # so that we can assign new stack pointer and the new base pointer
    1162:	sub    rsp,0x20                   # allocate 0x20 bytes for local vars
    # The above three instructions can be replaced with enter 0x20, 0. But enter is slow
    # so compilers generally don't use it. But compilers still use leave.
    1166:	mov    QWORD PTR [rbp-0x18],rdi   # rdi stores a/multiplicand
    116a:	mov    QWORD PTR [rbp-0x20],rsi   # rsi stores b/multiplier
    116e:	mov    QWORD PTR [rbp-0x8],0x0    # initialize local variable product
    1176:	mov    QWORD PTR [rbp-0x10],0x0   # it is unclear what *(rbp-0x10) stores until 1197. It turns out to be i
    117e:	jmp    119c <multiply+0x3e>
    1180:	mov    rdx,QWORD PTR [rbp-0x18]     
    1184:	mov    rax,QWORD PTR [rbp-0x8]       
    1188:	mov    rsi,rdx                    # rsi (2nd param in func call) stores a/multiplicand
    118b:	mov    rdi,rax                    # rdi (1st param) stores product
    118e:	call   1135 <add>                 # return address, i.e., 1193, is implicitly pushed to the call stack before jmp'ing to 1135
    1193:	mov    QWORD PTR [rbp-0x8],rax    # jmp'ed from 115d
    1197:	add    QWORD PTR [rbp-0x10],0x1
    119c:	mov    rax,QWORD PTR [rbp-0x10]       
    11a0:	cmp    rax,QWORD PTR [rbp-0x20]   # *(rbp-0x20) stores b/multiplier
    11a4:	jb     1180 <multiply+0x22>
    11a6:	mov    rax,QWORD PTR [rbp-0x8]    # prepare retval, product
    11aa:	leave                             # equivalent to mov rsp,rbp then pop rbp
    # comparing this to 115c, add() function does not change rsp, so it does not need to restore rbp.
    11ab:	ret    

00000000000011ac <main>:
    11ac:	push   rbp
    11ad:	mov    rbp,rsp
    11b0:	sub    rsp,0x20                    # Allocate 0x20 bytes to the stack frame of main()
    11b4:	mov    QWORD PTR [rbp-0x8],0xc     # *(rbp-0x8) = 12. rbp stores a memory address on the call stack, so dereferencing (rbp-0x8) gives us a memory block on the call stack
    11bc:	mov    QWORD PTR [rbp-0x10],0x22   # *(rbp-0x10) = 34
    11c4:	mov    rdx,QWORD PTR [rbp-0x10]    # load 34 to rdx
    11c8:	mov    rax,QWORD PTR [rbp-0x8]
    11cc:	mov    rsi,rdx                      
    11cf:	mov    rdi,rax                     # rdi and rsi are used to store first two parameters in function call
    11d2:	call   115e <multiply>             # return address, i.e., 11d7, is implicitly pushed to the call stack before jmp'ing to 115e
    11d7:	mov    QWORD PTR [rbp-0x18],rax    # retval (product)
    11db:	mov    rcx,QWORD PTR [rbp-0x18]
    11df:	mov    rdx,QWORD PTR [rbp-0x10]
    11e3:	mov    rax,QWORD PTR [rbp-0x8]
    11e7:	mov    rsi,rax
    11ea:	lea    rdi,[rip+0xe13]        # 2004 <_IO_stdin_used+0x4>
    11f1:	mov    eax,0x0
    11f6:	call   1030 <printf@plt>
    11fb:	mov    eax,0x0
    1200:	leave  
    1201:	ret    
    1202:	nop    WORD PTR cs:[rax+rax*1+0x0]
    120c:	nop    DWORD PTR [rax+0x0]
```
