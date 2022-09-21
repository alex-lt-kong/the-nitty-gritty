main.out: main.c
	gcc main.c -g -o main.out -O2
	gdb --quiet --eval-command="set disassembly-flavor intel" --eval-command="disassemble /m main" --batch ./main.out > ./main.asm

clear:
	rm *.out