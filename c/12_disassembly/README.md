# Disassembly

## x86-64 register fundamentals


### Purposes
* `rdi`, `rsi`, `rdx` and other three registers: store arguments of a function call. Additional arguments are
stored on the stack.
* `rax`: store return value of a function call.
* `rbp`: register base pointer, which points to the base of the current stack frame.
* `rsp`: register stack pointer, store the stack pointer.
* `xmm0`-`xmm15`: use by an SIMD instruction set to vectorize array operation, etc.

### Backward compatibility and their naming convention

* In x64 architecture, many 64-bit registers can be accessed independently as 32-, 16- or 8-bit registers.

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

* This design is also related to Intel's choice of using little-endian byte order. For example, if a 64-bit
register stores `0xDE AD BE EF 01 23 45 67` and we want to keep only its 32-bit part, it is more "reasonable" to
keep `0x01 23 45 67` instead of `0xDE AD BE EF`. If we want to achieve this with the above compatibility design,
we have to choose little-endian instead of big-endian byte order. That is, the first byte should store `0x67`
rather than `0xDE`.

### References

* [x64 Cheat Sheet](https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf)