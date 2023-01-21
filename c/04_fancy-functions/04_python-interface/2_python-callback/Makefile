OPTS = -O3 -Wall -pedantic -Wc++-compat -Wextra 
CC = gcc

main: func.c
	$(CC) -fPIC -shared -O3 -o func.so func.c -I/usr/include/python3.9/ -lpython3.9
