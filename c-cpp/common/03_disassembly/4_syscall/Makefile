main.out: main.c
	gcc main.c -g -o main.out -O1

.PHONY: clean disassemble
disassemble:
	objdump --disassembler-options "intel"  --no-show-raw-insn --source ./main.out > ./main.asm

clean:
	rm *.out