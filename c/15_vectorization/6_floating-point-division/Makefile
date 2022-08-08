main: main-flt.c main-int.c func.o utils.o
	gcc main-flt.c func-on.o utils.o -o main-flt-on.out -g -O3 -lm
	gcc main-flt.c func-no.o utils.o -o main-flt-no.out -g -O3 -lm -fno-tree-vectorize
	gcc main-int.c func-on.o utils.o -o main-int-on.out -g -O3 -lm
	gcc main-int.c func-no.o utils.o -o main-int-no.out -g -O3 -lm -fno-tree-vectorize

utils.o: ../utils.c ../utils.h
	gcc -c ../utils.c -O2

func.o: func.c func.h
	gcc -c func.c -o func-on.o -g -O3
	gcc -c func.c -o func-no.o -g -O3 -fno-tree-vectorize -mno-avx

clear:
	rm *.out *.o
