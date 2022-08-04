main: main.c func.o utils.o	
	icc main.c func-icc-on.o utils.o -o main-icc-on.out -g -O3 -lm 
	icc main.c func-icc-no.o utils.o -o main-icc-no.out -g -O3 -lm -no-vec
	gcc main.c func-gcc-on.o utils.o -o main-gcc-on.out -g -O3 -lm
	gcc main.c func-gcc-no.o utils.o -o main-gcc-no.out -g -O3 -lm -fno-tree-vectorize

utils.o: ../utils.c ../utils.h
	icc -fpic -c ../utils.c -O3 -fpic

func.o: func.c func.h
	icc -c func.c -o func-icc-on.o -g -O3
	icc -c func.c -o func-icc-no.o -g -O3 -no-vec
	gcc -c func.c -o func-gcc-on.o -g -O3
	gcc -c func.c -o func-gcc-no.o -g -O3 -fno-tree-vectorize

clear:
	rm *.out *.o
