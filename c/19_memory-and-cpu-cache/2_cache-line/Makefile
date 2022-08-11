main.out: main.c
	gcc main.c -o main-gcc.out -O3 -fno-tree-vectorize
	icc main.c -o main-icc.out -O3 -no-vec
	gdb -batch -ex 'file ./main-gcc.out' -ex 'disassemble /m main' > ./main-gcc.asm
	gdb -batch -ex 'file ./main-icc.out' -ex 'disassemble /m main' > ./main-icc.asm
	lscpu > ./system.env
clean:
	rm *.out