main.out: main.c
	gcc main.c -o main-gcc.out -O3 -Wall
	icc main.c -o main-icc.out -O3 -Wall
	lscpu > ./lscpu.txt
clean:
	rm *.out
