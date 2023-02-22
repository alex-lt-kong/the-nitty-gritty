* Compile with the `-g` flag to include debug information in the binary, making it possible to inspect it using GDB: `gcc faulty.c -Og -g -o faulty`

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
