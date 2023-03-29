# Disassembly

To dissamble a binary file to assembly code:

* Dissamble the entire binary file:
    * `objdump --disassembler-options "intel" -S <binary-file>`
* Dissamble only a function from a binary file:
    * `gdb`: `gdb --quiet --eval-command="set disassembly-flavor intel" --eval-command="disassemble /m <func-name>" --batch <binary_file>`.
    * `objdump`: `objdump --disassembler-options "intel" [--demangle] [--no-show-raw-insn] --disassemble=<func-name> --source <binary_file>`.
        * `--demangle`: for C++, it is used to decode (a.k.a. demangle)
        low-level symbol names into user-level human-friendly names.
            * Name mangling is the encoding of function and variable names
            into unique names so that linkers can separate common names in
            the language. Name mangling is commonly used to facilitate
            the overloading feature and visibility within different scopes.
        * `--no-show-raw-insn`: Disable the output of instruction bytes.
* Dump content by address: `objdump --full-content --start-address=<start> --stop-address=<stop> <binary-file>`
* Dump (not disassemble) the data sections:
    1. Check how many sections are there:
    `readelf --sections --wide <binary-file>`, we are most likely to be
    interested in the `.text` and `.rodata` sections:
        ```
            Section Headers:
            [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
        ...
            [10] .rela.plt         RELA            00000000000007d8 0007d8 000090 18  AI  5  24  8
            [11] .init             PROGBITS        0000000000001000 001000 000017 00  AX  0   0  4
            [12] .plt              PROGBITS        0000000000001020 001020 000070 10  AX  0   0 16
            [13] .plt.got          PROGBITS        0000000000001090 001090 000008 08  AX  0   0  8
            [14] .text             PROGBITS        00000000000010a0 0010a0 0001f1 00  AX  0   0 16
            [15] .fini             PROGBITS        0000000000001294 001294 000009 00  AX  0   0  4
            [16] .rodata           PROGBITS        0000000000002000 002000 000032 00   A  0   0  8
            [17] .eh_frame_hdr     PROGBITS        0000000000002034 002034 00004c 00   A  0   0  4
            ...
        ```
    1. Dump the content, without "disassemble" it: 
        `readelf --hex-dump .rodata <binary-file>` or
        `objdump --full-content --section .rodata <binary-file>`
        ```
            Contents of section .rodata:
            2000 01000200 00000000 54686973 20697320  ........This is
            2010 61207665 7279206c 6f6f6f6f 6f6f6f6f  a very loooooooo
            2020 6f6f6f6f 6f6f6f6f 6f6f6f6f 6f6f6f6f  oooooooooooooooo
            2030 6f6f6f6f 6e672070 72656669 78207468  oooong prefix th
            2040 61742064 6f65736e 27742066 69742069  at doesn't fit i
            2050 6e746f20 32332062 79746573 00000000  nto 23 bytes....
            2060 62617369 635f7374 72696e67 3a3a5f4d  basic_string::_M
            2070 5f636f6e 73747275 6374206e 756c6c20  _construct null
            2080 6e6f7420 76616c69 64000000 00000000  not valid.......
            2090 48656c6c 6f2c2077 6f726c64 21205468  Hello, world! Th
            20a0 69732073 7472696e 67206973 206c6f6e  is string is lon
            20b0 67657220 7468616e 2031312f 32332063  ger than 11/23 c
            20c0 68617261 63746572 732e006c 6f6e6753  haracters..longS
            20d0 7472696e 673a2000                    tring: .
        ```
    1. But what if we try to disassemble the `.rodata` section? Say, we run: 
    `objdump --disassembler-options "intel" --disassemble --no-show-raw-insn --section .rodata.str1.8 <binary-file>`?
        * `objdump` will disassemble it to legit-looking non-sense:
        ```
        0000000000000000 <.LC0>:
            0:   push   rsp
            1:   push   0x69207369
            6:   jae    28 <.LC0+0x28>
            8:   (bad)
            9:   and    BYTE PTR [rsi+0x65],dh
            c:   jb     87 <.LC0+0x87>
            e:   and    BYTE PTR [rdi+rbp*2+0x6f],ch
            12:   outs   dx,DWORD PTR ds:[rsi]
            13:   outs   dx,DWORD PTR ds:[rsi]
            14:   outs   dx,DWORD PTR ds:[rsi]
            15:   outs   dx,DWORD PTR ds:[rsi]
            16:   outs   dx,DWORD PTR ds:[rsi]
            17:   outs   dx,DWORD PTR ds:[rsi]
            18:   outs   dx,DWORD PTR ds:[rsi]
            19:   outs   dx,DWORD PTR ds:[rsi]
            1a:   outs   dx,DWORD PTR ds:[rsi]
            1b:   outs   dx,DWORD PTR ds:[rsi]
            1c:   outs   dx,DWORD PTR ds:[rsi]
            1d:   outs   dx,DWORD PTR ds:[rsi]
            1e:   outs   dx,DWORD PTR ds:[rsi]
            1f:   outs   dx,DWORD PTR ds:[rsi]
            20:   outs   dx,DWORD PTR ds:[rsi]
            21:   outs   dx,DWORD PTR ds:[rsi]
            22:   outs   dx,DWORD PTR ds:[rsi]
            23:   outs   dx,DWORD PTR ds:[rsi]
            24:   outs   dx,DWORD PTR ds:[rsi]
            25:   outs   dx,DWORD PTR ds:[rsi]
            26:   outs   dx,DWORD PTR ds:[rsi]
            27:   outs   dx,DWORD PTR ds:[rsi]
            28:   outs   dx,DWORD PTR ds:[rsi]
            29:   outs   dx,DWORD PTR ds:[rsi]
            2a:   outs   dx,DWORD PTR ds:[rsi]
            2b:   outs   dx,DWORD PTR ds:[rsi]
            2c:   outs   dx,BYTE PTR ds:[rsi]
            2d:   and    BYTE PTR [eax+0x72],dh
            31:   imul   di,WORD PTR gs:[rax+0x20],0x6874
            38:   (bad)
            39:   je     5b <.LC0+0x5b>
            3b:   outs   dx,DWORD PTR fs:[rsi]
            3d:   gs jae ae <.LC0+0xae>
            40:   (bad)
            41:   je     63 <.LC0+0x63>
            43:   imul   si,WORD PTR [rax+riz*1+0x69],0x746e
            4a:   outs   dx,DWORD PTR ds:[rsi]
            4b:   and    BYTE PTR [rdx],dh
            4d:   xor    esp,DWORD PTR [rax]
            4f:   (bad)
            50:   jns    c6 <.LC0+0xc6>
            52:   gs jae 55 <.LC0+0x55>
        ``` 



## x86-64 register fundamentals

### Purposes
* `rdi`, `rsi`, `rdx`, `rcx`, `r8` and `r9`: pass first 6 integer/pointer
arguments of a function call. Additional arguments are stored on the call stack.
    * `xmm0` - `xmm7`: among other general purpose usage, can be used to pass
    first 8 float pointing arguments. Additional arguments are stored on the
    stack.
* `rax`: store return value of a function call.
* `rbp`/`ebp`: register base pointer, a.k.a., frame pointer, which points to
the base of the "current" stack frame.
    * At a higher level, it means that all variables local to a function
    is stored "after" the memory address stored by it.
    * As the call stack grows down, it means `[rbp-0x18]` could store a
    local variable, but `[rbp+0x18]` could never store a local variable as
    it is "before" the current stack frame, most likely belong to the previous
    stack frame, meaning that is could be a variable local to the caller
    of current function.
* <a id="call-stack-pointer">`rsp`/`esp`</a>: stores the call stack pointer, which
points to the "top" of the stack. Note that call stack usually grows downward.
Therefore, while conceptually being the "top" of the call stack, `rsp` has the
smallest address value in the stack.
* `xmm0`-`xmm15`: use by an SIMD instruction set to vectorize array
operation, etc.
* `rip`/`eip` is a 64bit/32bit register. It holds the "Extended Instruction
Pointer" for the call stack. In other words, it tells the CPU where to go
next to execute the next command. Behind the scene, `call` and `jmp`
both change the value of this register.
    * To be more concrete, `rip`/`eip` stores the address of the instruction
    immediately following the current instruction.
    * In the below example, rip stores `11d6`:
    ```asm
    11cc:	mov    rdi,rsp
    11cf:	lea    rsi,[rip+0xe32]
    11d6:	call   1050
    11db:	mov    rdi,QWORD PTR [rsp]
    ```

### Backward compatibility and their naming convention

* In x64 architecture, many 64-bit registers can be accessed independently as
32-, 16- or 8-bit registers.

* This is also the approach Intel takes to maintain backward compatibility.

| 64-bit register | 0-31 bits   | 0-15 bits  | 0-7 bits   |
| --------------- | ----------- |----------- |----------- |
| rax             | eax         | ax         | al         |
| rbx             | ebx         | bx         | bl         |
| rcx             | ecx         | cx         | cl         |
| rdx             | edx         | dx         | dl         |
| rdi             | edi         | di         | dil        |
| rsi             | esi         | si         | sil        |
| rsp             | esp         | sp         | spl        |
| rbp             | ebp         |            |            |

* The `e` in `eax`,`ebx`,etc compared with `ax`, `bx` means "extended",
meaning that 16-bit registers are "extended" to 32 bits.

* This design is also related to Intel's choice of using little-endian byte
order. For example, if a 64-bit register stores `0xDE AD BE EF 01 23 45 67`
and we want to keep only its 32-bit part, it is more "reasonable" to keep
`0x01 23 45 67` instead of `0xDE AD BE EF`. If we want to achieve this with
the above compatibility design, we have to choose little-endian instead of
big-endian byte order. That is, the first byte should store `0x67` rather
than `0xDE`.

### Notes

* In a typical assembly line, `opcode operand1, operand2`, which is source and
which is destination? Let's use `mov` as an example.
    * `mov dst, src` is called Intel syntax. (e.g. mov eax, 123)
    * `mov src, dst` is called AT&T syntax. (e.g. mov $123, %eax)


* Caller-saved and callee-saved registers

    * Caller-saved registers (a.k.a. volatile registers, or call-clobbered) are
    used to hold temporary quantities that need not be preserved across calls.
    * Callee-saved registers (a.k.a. non-volatile registers, or call-preserved)
    are used to hold long-lived values that should be preserved across calls.
    * The so-called "callee-saved" means that the callee has to save the
    registers and then restore them at the end of the call because they have
    the guarantee to the caller of containing the same values after the
    function returns. The usual way to save the callee-saved registers is
    to push them onto the call stack.

## Making sense of some common operations

* `xor eax,eax`: `xor`ing/`pxor`ing a register with itself is a faster way
of setting the register to zero.

* `sub rsp,0x88`: `esp`/`rsp` is the register stack pointer pointing to the
"top" of the call stack. `sub`tracting `0x88` from `rsp` means we allocate
`0x88` bytes to the new stack frame, i.e., to be used to store function-
specific variables.

* `mov [ebx],eax`: it roughly means `*ebx = eax`, i.e., moves the value in
`eax` to the memory address contained in `ebx`.

* `imul rax,rbx,0x16`: `imul` is signed multiplication. The less common part
is that it has three operands. It means `rax = rbx * 0x16`.

* `jmp rax`: it actually means `jmp [rax]`.

* `mov edx, [ebx + 8*eax + 4]` and `lea edx, [ebx + 8*eax + 4]`:
    * Say we have a struct:

    ```C
    struct Point {
        int xcoord;
        int ycoord;
    };
    ```

        and an array of `Point points[]`.
    * In C we do `int y = points[i].ycoord;`, which could be translated to
    `mov edx, [ebx + 8*eax + 4]` if `ebx` stores the base pointer `&points[0]`
    and `eax` stores `i`. We have `8*eax` because each `Point` is 8-byte long.
    * In C we do `int* ptr = &(points[i].ycoord);`, which could be translated to
    `lea edx, [ebx + 8*eax + 4]`.

* `pop`/`push`: they change [`esp`/`rsp`](#call-stack-pointer) (call stack
pointer) implicitly.
    * `pop esi` is roughly the same as:

    ```asm
    mov esi, [esp]
    add esp, 4  # for x86; 8 for x64
    ```

    * `push esi` roughly means:

    ```asm
    sub esp, 4   # for x86; 8 for x64
    mov [esp], esi 
    ```

    * As stack stores data from the top down, we perform `add` for `pop` and
    `sub` for `push`.

* `call`/`ret`: the way `call` and `ret` work is called the "calling
convention". An informative video can be found
[here](./0_assets/x86-calling-convention.mp4). We are going to describe
how they work with a toy example extracted from [here](./5_function-call/),
where `call` is executed at `118e` and `ret` is executed at `115d`:

    ```asm
    ...
    1135:	push   rbp                       # previous base pointer's address is pushed to the call stack...
    1136:	mov    rbp,rsp                   # so that we can assign new stack pointer and the new base pointer
    1139:	mov    QWORD PTR [rbp-0x18],rdi  # a/product
    113d:	mov    QWORD PTR [rbp-0x20],rsi  # b/multiplicand
    1141:	mov    QWORD PTR [rbp-0x8],0x0   # sum    
    1158:	mov    rax,QWORD PTR [rbp-0x8]   # rax stores return value
    115c:	pop    rbp                      
    115d:	ret                              # pop's another element from the call stack, i.e., the return address, which was push'ed to the call stack at 
    ...
    1188:	mov    rsi,rdx                   # rsi (2nd param in func call) stores a/multiplicand
    118b:	mov    rdi,rax                   # rdi (1st param) stores product
    118e:	call   1135 <add>                # return address, i.e., 1193, is implicitly pushed to the call stack before jmp'ing to 1135
    1193:	mov    QWORD PTR [rbp-0x8],rax   # jmp'ed from 115d
    ...
    ```

    * `call` instruction does the following things:
        1. it `push`es the return address (i.e., the address of
        the instruction immediately after the `call` instruction. In the
        example, the address being `push`ed is `1193`) to the call stack.
        1. it `jmp`s to the address of being called. In
        the above example, `1135`. Internally, it sets the `eip` register to
        `1135`.        
        1. It is equivalent to:
        ```asm
        push 1193
        jmp 1135
        ```
        * Note that `call` instruction only saves return address (e.g., `1193`
        ) to the call stack but it does not create a new stack frame. The new stack
        frame is created by the callee itself at `1135` and `1136`. More one
        stack frame set up will be explained in the next section (about
        `enter` and `leave)
    * `ret` is the reverse of `call`:
        ```asm
        pop rcx # rcx is a general-purpose register we pick at random
        jmp [rcx]
        ```
        * Note that during the "function call" we `push`ed twice. The 1st `push`
        is implicitly invoked by `call`, which stores the return address to the
        stack. The 2nd `push` is explicitly executed in the callee function (at
        `1135`), storing the base of the previous stack frame to the call stack.
        * Similarly, `ret` involves two `pop`s.
            1. The 1st `pop` is explicitly executed at `115d`.
            1. The 2nd one is done implicitly by `ret`, which `pop`s the return
            address (in this case, `1193`) from stack and `jmp`s to it.
    * In this example, we dont have `enter` and `leave`, probably due to the
    fact that add() is at the bottom of the call stack, so that we don't need
    to further delimit the call stack frame for subsequent calls.

* `enter`/`leave`: they are closely related to `call`/`ret`.
    * We use the following toy example from [here](./5_function-call/) to
    demonstrate what they do:
        ```asm
        115e:	push   rbp                        # previous base pointer's address is pushed to the call stack...
        115f:	mov    rbp,rsp                    # so that we can assign new stack pointer and the new base pointer
        1162:	sub    rsp,0x20                   # allocate 0x20 bytes for local vars
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
        11aa:	leave
        11ab:	ret    
        ```
    * The `enter` instruction sets up a stack frame by first pushing
    `rbp` onto the call stack and then copies `rsp` into `rbp` (`115e`
    and `115f`).
    * `leave` does the opposite, i.e. copy `rbp` to `rsp` and then restore
    the old `rbp` from the call stack:
        ```asm
        mov  rsp, rbp 
        pop  rbp
        ```
    * Note that `enter` instruction is very slow and compilers don't use it,
    but `leave` is fine.
    * Comparing with the example in `call`/`ret`, one may notice that in this
    example, `rsp` is changed to allocate 0x20 bytes so we need to `leave`
    to restore it from the call stack. In the previous example, `rsp` is not
    changed, so `leave` is not executed anb we just `pop` instead.
    * Note that call stack "grows downward", meaning that the direction to
    "enlarge" stack is negative.
        * For example, `sub    rsp,0x20` "grows" the stack by moving stack top
        pointer "down". By doing so we allocate extra `0x20` bytes of space
        to the stack.
        * However, heap does the opposite, it grows "upward".

### References

* [x64 Cheat Sheet](https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf)

* [x86 instruction reference](https://www.felixcloutier.com/x86/): `https://www.felixcloutier.com/x86/<opcode here>`
