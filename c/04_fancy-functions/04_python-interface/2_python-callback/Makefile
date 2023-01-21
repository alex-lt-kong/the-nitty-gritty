OPTS = -O3 -Wall -pedantic -Wc++-compat -Wextra 
CC = clang

main: func.c
	$(CC) -fPIC -shared -O3 -o func.so func.c

clean:
	rm *.so