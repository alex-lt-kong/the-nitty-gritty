main: main-flt.c main-int.c func.o utils.o
	gcc main-flt.c func-on.o utils.o -o main-flt-on.out -g -O3 -lm
	gcc main-flt.c func-no.o utils.o -o main-flt-no.out -g -O3 -lm -fno-tree-vectorize
	gcc main-int.c func-on.o utils.o -o main-int-on.out -g -O3 -lm
	gcc main-int.c func-no.o utils.o -o main-int-no.out -g -O3 -lm -fno-tree-vectorize

utils.o: ../utils.c ../utils.h
	gcc -c ../utils.c -O2

func.o: func.c func.h
	gcc -c func.c -o func-on.o -g -O3
	gdb --quiet --eval-command="disassemble /m func_floating_division" --eval-command="disassemble /m func_int_multiplication" --batch ./func-on.o | tail -n +2 | head -n -1 | cut -c 4- > ./func-on.asm

	gcc -c func.c -o func-no.o -g -O3 -fno-tree-vectorize -mno-avx
	gdb --quiet --eval-command="disassemble /m func_floating_division" --eval-command="disassemble /m func_int_multiplication" --batch ./func-no.o | tail -n +2 | head -n -1 | cut -c 4- > ./func-no.asm
	
clear:
	rm *.out *.o
