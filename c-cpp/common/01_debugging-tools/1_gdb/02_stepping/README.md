* Example is from the [disassembly directory](../../disassembly)
* **Prepare programs:**
    * Embed debugging info into executables: `gcc main.c -Og -g -o main.out`.

* **Examining variables**:
    * print variable value at breakpoint: `print [var name]`.
    * print register value: `info registers` to print all or `info registers [reg name]` to print one.

* **Breakpoint**:
    * A breakpoint "at" a line means we stop the exeuction before that line.
    * Add a breakpoint at source code level: `break [file name]:[line no]` (e.g. `break func.c:1`) then `run`.
    * List existing breakpoints: `info break`.
    * Disable a breakpoint: `disable breakpoint [breakpoint num]`.
    * Delete a breakpoint: `del breakpoint [breakpoint num]`.
    * Enable step mode: `set step-mode on` then `run`.
    * Check step mode status: `show step-mode`.

* **Assembly/Disassembly**
    * Show all assembly instructions: `layout asm`, or assembly instructions and source code side-by-side: `layout split`.nq
    * Set assembly syntax to "Intel": `set disassembly-flavor intel`.
    * Show current assembly instructions: `set disassemble-next-line on` then `show disassemble-next-line`
    * Disassemble one single function from one file: `gdb -batch -ex 'file <file_path>' -ex 'disassemble <function_name>'`
    * Disassemble one single function from one file with original C lines prepended:
    `gdb -batch -ex 'file ./main-icc-on.out' -ex 'disassemble /m linear_func_uint32' | cut -d " " -f 5-`

* **Step-by-step execution**
    * `n` for execution per source code line.
    * `ni` for execution per machine code instruction.

* **Useful links**:
    * [Debugging with GDB](https://sourceware.org/gdb/download/onlinedocs/gdb/)