main: main.c utils.o
	gcc main.c utils.o -o main-gcc-on.out -g -O3 -lm    -ftree-vectorize
	gcc main.c utils.o -o main-gcc-no.out -g -O3 -lm -fno-tree-vectorize
	icc main.c utils.o -o main-icc-on.out -g -O3 -lm    -vec -qopt-report -qopt-report-phase=vec
	icc main.c utils.o -o main-icc-no.out -g -O3 -lm -no-vec

utils.o: ../utils.c ../utils.h
	gcc -c ../utils.c -O3

clear:
	rm *.out *.o