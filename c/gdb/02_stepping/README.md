* Example is from the [disassembly directory](../../disassembly)
* Embed debugging info into executables: `gcc main.c -Og -g -o main.out`
* Add a breakpoint at source code level: `break 5` then `run`
* print variable value at breakpoint: `print [var name]`
* Enable stepping mode: `set step-mode on` then `run`
* print register value: `info registers` to print all or `info registers [reg name]` to print one