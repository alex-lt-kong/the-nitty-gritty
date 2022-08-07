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


### References

* [x64 Cheat Sheet](https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf)