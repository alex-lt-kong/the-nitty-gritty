main: main-aos.c main-soa.c func.o utils.o
	icc main-aos.c func-on.o utils.o -o main-aos-on.out -g -O3 -lm
	icc main-aos.c func-no.o utils.o -o main-aos-no.out -g -O3 -lm -fno-tree-vectorize
	icc main-soa.c func-on.o utils.o -o main-soa-on.out -g -O3 -lm
	icc main-soa.c func-no.o utils.o -o main-soa-no.out -g -O3 -lm -fno-tree-vectorize

utils.o: ../utils.c ../utils.h
	icc -c ../utils.c -O2

func.o: func.c func.h
	icc -c func.c -o func-on.o -g
	icc -c func.c -o func-no.o -g -ftree-vectorize

clear:
	rm *.out *.o
