main: main.c func.o
	gcc main.c func.o -o main.out -g -O3

func.o: func.c func.h
	gcc -c func.c -o func.o -g -O3 -fopt-info-vec-missed

clean:
	rm *.out *.o
