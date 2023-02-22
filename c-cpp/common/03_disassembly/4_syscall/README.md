# Syscall

* In computing, a system call (commonly abbreviated to syscall) is the programmatic way in which a computer
program requests a service from the kernel of the operating system on which it is executed. This may include
hardware-related services (for example, accessing a hard disk drive or accessing the device's camera), creation and
execution of new processes, and communication with integral kernel services such as process scheduling.

* In x64/32, parameters for Linux `syscall` are passed using registers: `%rax`/`%eax` for syscall_number,
`%rbx`/`%ebx`, `%rcx`/`%ecx`, `%rdx`/`%edx`, `%rsi`/`%esi`, `%rdi`/`%edi`, `%rbp`/`%ebp` are used for passing
6 parameters to system calls. The return value is in `%rax`/`%eax`.

* In this simple `main.c`, `open`, `read` and `close` are all `syscall`'s. The question is, how does the compiler know
that they are `syscall`s? The [answer](https://stackoverflow.com/questions/3546760/how-does-compiler-know-that-the-function-you-used-is-a-system-call)