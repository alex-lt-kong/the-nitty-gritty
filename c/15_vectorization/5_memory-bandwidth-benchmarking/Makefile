main: main.c utils.o
	gcc main.c utils.o -o main.out -O2 -lm

utils.o: ../utils.c ../utils.h
	gcc -c ../utils.c -O2