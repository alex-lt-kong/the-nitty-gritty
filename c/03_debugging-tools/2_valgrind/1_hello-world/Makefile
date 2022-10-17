main: main.c
	gcc -o main.out main.c -O1 -Wall
	valgrind --leak-check=yes --log-file=valgrind.rpt ./main.out