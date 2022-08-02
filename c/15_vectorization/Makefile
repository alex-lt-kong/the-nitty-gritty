main: main.c simple-loop.o
	gcc main.c simple-loop-gcc-on.o -o main-gcc-on.out -O3    -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed
	gcc main.c simple-loop-gcc-no.o -o main-gcc-no.out -O3 -fno-tree-vectorize
	icc main.c simple-loop-icc-on.o -o main-icc-on.out -O2 -vec
	icc main.c simple-loop-icc-no.o -o main-icc-no.out -O2 -no-vec

simple-loop.o: simple-loop.c simple-loop.h
	gcc simple-loop.c -c -o simple-loop-gcc-on.o -O2    -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed
	gcc simple-loop.c -c -o simple-loop-gcc-no.o -O2 -fno-tree-vectorize
	icc simple-loop.c -c -o simple-loop-icc-on.o -O2 -vec
	icc simple-loop.c -c -o simple-loop-icc-no.o -O2 -no-vec 

clear:
	rm *.out *.o