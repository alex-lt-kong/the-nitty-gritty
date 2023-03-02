main.out: main.c
	gcc main.c -fno-inline -g -o main.out
	objdump --disassembler-options "intel" --source ./main.out --no-show-raw-insn > ./main.asm

.PHONY: clean
clean:
	rm *.out