main: main-aos.c main-soa.c func.o utils.o
	gcc main-aos.c func-on.o utils.o -o main-aos-on.out -g -O3 -lm
	gcc main-aos.c func-no.o utils.o -o main-aos-no.out -g -O3 -lm -fno-tree-vectorize
	gcc main-soa.c func-on.o utils.o -o main-soa-on.out -g -O3 -lm
	gcc main-soa.c func-no.o utils.o -o main-soa-no.out -g -O3 -lm -fno-tree-vectorize

utils.o: ../utils.c ../utils.h
	gcc -c ../utils.c -O2

func.o: func.c func.h
	gcc -c func.c -o func-on.o -g
	gcc -c func.c -o func-no.o -g -ftree-vectorize

clear:
	rm *.out *.o
