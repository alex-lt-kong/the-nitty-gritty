main.out: main.c
	gcc main.c -o main.out
	gcc -c -ggdb sum.c
	gdb --quiet --eval-command="disassemble sum" --batch ./sum.o | tail -n +2 | head -n -1 | cut -c 4- > sum.asm
	./main.out
clear:
	rm *.o
	rm *.out