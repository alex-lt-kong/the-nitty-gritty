main.out: main.c
	gcc main.c -fno-inline -g -o main.out
	gcc main.c -fno-inline -g -o main-o2.out -O2
	objdump --disassembler-options "intel" --source ./main.out --no-show-raw-insn > ./main.asm
	objdump --disassembler-options "intel" --source ./main-o2.out --no-show-raw-insn > ./main-o2.asm

.PHONY:
clean:
	rm *.out