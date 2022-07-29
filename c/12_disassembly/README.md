* Useful online disassembler with annotation: `https://godbolt.org/`, `http://www.ctoassembly.com/`.
* Disassemble executables with `gdb`: `gdb --quiet --eval-command="set disassembly-flavor intel" --eval-command="disassemble sum_them_all" --batch ./func.o | tail -n +2 | head -n -1 | cut -c 4-`
* Some more "complicated" cases are dynamically analyzed in the [gdb directory](../gdb/)