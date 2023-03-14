# GDB fundamentals

* settings file: `.gdbinit`. (For example, to use Intel's flavor by default, add `set disassembly-flavor intel` to it)

## Basic usage

* Code is in [01_basic](./01_basic/)

* Compile with the `-g` flag to include debug information in the binary,
making it possible to inspect it using GDB: `gcc faulty.c -Og -g -o faulty`

* Normal run:
```
# ./faulty 
Segmentation fault
```
* Run with `GDB`:
```
# gdb ./faulty 
GNU gdb (Debian 10.1-1.7) 10.1.90.20210103-git
Copyright (C) 2021 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./faulty...
(gdb) run
Starting program: /mnt/hdd0/repos/the-nitty-gritty/c_cpp/gdb/faulty 

Program received signal SIGSEGV, Segmentation fault.
__strlen_avx2 () at ../sysdeps/x86_64/multiarch/strlen-avx2.S:65
65	../sysdeps/x86_64/multiarch/strlen-avx2.S: No such file or directory.
```

## Common settings and options

* Example is from the [disassembly directory](../../../03_disassembly/)


### Prepare programs

* Embed debugging info into executables: `gcc main.c -Og -g -o main.out`.


### Examining variables

* print variable value at breakpoint: `print [var name]`.
* print register value: `info registers` to print all or
`info registers [reg name]` to print one.
* print value at a memory address: `x /n addr` (e.g., `x /4 0x123456`)


### Breakpoint

* A breakpoint "at" a line means we stop the exeuction before that line.
* Add a breakpoint at source code level: `break [file name]:[line no]`
(e.g., `break func.c:1`)
* List existing breakpoints: `info break`.
* Disable a breakpoint: `disable breakpoint [breakpoint num]`.
* Delete a breakpoint: `del breakpoint [breakpoint num]`.
* Enable step mode: `set step-mode on` then `run`.
* Check step mode status: `show step-mode`.


### Assembly/Disassembly

* Show all assembly instructions: `layout asm`, or assembly instructions
and source code side-by-side: `layout split`.nq
* Set assembly syntax to "Intel": `set disassembly-flavor intel`.
* Show current assembly instructions: `set disassemble-next-line on` then
`show disassemble-next-line`
* Disassemble one single function from one file:
`gdb -batch -ex 'file <file_path>' -ex 'disassemble <function_name>'`
* Disassemble one single function from one file with original C lines prepended:
`gdb -batch -ex 'file ./main-icc-on.out' -ex 'disassemble /m linear_func_uint32' | cut -d " " -f 5-`


### Step-by-step execution

* `n` for execution per source code line.
* `ni` for execution per machine code instruction.


### Useful links

* [Debugging with GDB](https://sourceware.org/gdb/download/onlinedocs/gdb/)