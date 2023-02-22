main: test.c utils.o
	gcc test.c utils.o -o test.out -g -O2 -lm
	

utils.o: utils.c utils.h
	gcc -c utils.c

clear:
	rm *.out *.o