main.out: main.c
	gcc main.c -o main.out -O3 -fno-tree-vectorize
	gdb -batch -ex 'file ./main.out' -ex 'disassemble /m main' > ./main.asm
	lscpu > ./system.env
clean:
	rm *.out