# Disassembly

To dissamble a binary file to assembly code:

* Dissamble the entire binary file:
    * `objdump --disassembler-options "intel" -S ./main.out`
* Dissamble only a function from a binary file:
    * `gdb`: `gdb --quiet --eval-command="set disassembly-flavor intel" --eval-command="disassemble /m <func name>" --batch <exe name>`.
    * `objdump`: `objdump --disassembler-options "intel" --disassemble=<func name> -S <exe name>`.
* For C++, add `--demangle` to `objdump` to decode (a.k.a. demangle) low-level
symbol names into user-level
human-friendly names.
    * Name mangling is the encoding of function and variable names into unique
    names so that linkers can separate common names in the language. Name
    mangling is commonly used to facilitate the overloading feature and
    visibility within different scopes.

## x86-64 register fundamentals

### Purposes
* `rdi`, `rsi`, `rdx`, `rcx`, `r8` and `r9`: pass first 6 integer/pointer
arguments of a function call. Additional arguments are stored on the stack.
    * `xmm0` - `xmm7`: among other general purpose usage, can be used to pass
    first 8 float pointing arguments. Additional arguments are stored on the
    stack.
* `rax`: store return value of a function call.
* `rbp`/`ebp`: register base pointer, which points to the base of the current
stack frame.
* `rsp`/`esp`: register stack pointer, store the stack pointer.
* `xmm0`-`xmm15`: use by an SIMD instruction set to vectorize array operation, etc.
* `rip`, `eip` is a 64bit/32bit register. It holds the "Extended Instruction
Pointer" for the stack. In other words, it tells the CPU where to go next to
execute the next command. Behind the scene, `call` and `jmp` both change the
value of this register.

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
    to push them onto the stack.

## Making sense of some common operations

* `xor eax,eax`: `xor`ing/`pxor`ing a register with itself is a faster way
of setting the register to zero.

* `sub rsp,0x88`: `esp`/`rsp` is the register stack pointer pointing to the
"top" of the call stack. `sub`tracting `0x88` from `rsp` means we allocate
`0x88` bytes to the new stack frame, i.e., to be used to store function-
specific variables.

* `mov [ebx],eax`: it roughly means `*ebx = eax`, i.e., moves the value in
`eax` to the memory address contained in `ebx`.

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

* `pop`/`push`: they change `esp`/`rsp` implicitly.
    * `pop esi` is roughly the same as:

    ```asm
    mov esi, [esp]
    add esp, 4  ; for x86; 8 for x64
    ```

    * `push esi` roughly means:

    ```asm
    sub esp, 4   ; for x86; 8 for x64
    mov [esp], esi 
    ```

    * As stack stores data from the top down, we perform `add` for `pop` and
    `sub` for `push`.

* `call`/`ret`: the way `call` and `ret` work is called the "calling
convention". An informative video can be found
[here](./0_assets/x86-calling-convention.mp4). We are going to describe
how they work with a toy example extracted from [here](./5_function-call/),
where `call` is executed at `0x1166` and `ret` is executed at `0x1145`:
    ```asm
    1125:       55                      push   rbp
    1126:       48 89 e5                mov    rbp,rsp
    1129:       89 7d ec                mov    DWORD PTR [rbp-0x14],edi
    112c:       89 75 e8                mov    DWORD PTR [rbp-0x18],esi
    112f:       c7 45 fc 00 00 00 00    mov    DWORD PTR [rbp-0x4],0x0
    1136:       8b 55 ec                mov    edx,DWORD PTR [rbp-0x14]
    1139:       8b 45 e8                mov    eax,DWORD PTR [rbp-0x18]
    113c:       01 d0                   add    eax,edx
    113e:       89 45 fc                mov    DWORD PTR [rbp-0x4],eax
    1141:       8b 45 fc                mov    eax,DWORD PTR [rbp-0x4]
    1144:       5d                      pop    rbp
    1145:       c3                      ret
    ...
    1162:       89 d6                   mov    esi,edx
    1164:       89 c7                   mov    edi,eax
    1166:       e8 ba ff ff ff          call   1125 <add>
    116b:       89 45 f4                mov    DWORD PTR [rbp-0xc],eax
    116e:       8b 45 f4                mov    eax,DWORD PTR [rbp-0xc]
    ```.
    * `call` instruction does the following things:
        1. it `push`es the return address (i.e., the address of
        the instruction immediately after the `call` instruction. In the
        example, the address being `push`ed is `0x116b`) to the stack.
        1. it `jmp`s to the address of being called. In
        the above example, `0x1125`. Internally, it sets the `eip` register to
        `0x1125`.
        * Note that `call` instruction only saves return address (e.g., `0x116b`
        ) to the stack but it does not create a new stack frame. The new stack
        frame is created by the callee itself at `0x1125` and `0x1126`.
    * `ret` is the reverse of `call`.
        * Note that during the "function call" we `push`ed twice. The 1st `push`
        is implicitly invoked by `call`, which stores the return address to the
        stack. The 2nd `push` is explicitly executed in the callee function (at
        `0x1125`), storing the base of the previous stack frame to the stack.
        * Similarly, `ret` involves two `pop`s.
            1. The 1st `pop` is explicitly executed at `0x1144`.
            1. The 2nd one is done implicitly by `ret`, which `pop`s the return
            address (in this case, `0x116b`) from stack and `jmp`s to it.

### References

* [x64 Cheat Sheet](https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf)

* [x86 instruction reference](https://www.felixcloutier.com/x86/): `https://www.felixcloutier.com/x86/<opcode here>`
