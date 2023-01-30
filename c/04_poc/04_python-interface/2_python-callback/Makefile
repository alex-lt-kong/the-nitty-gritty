OPTS = -O3 -Wall -pedantic -Wc++-compat -Wextra 
CC = clang

main:
	$(CC) -fPIC -shared -o func.so func.c $(OPTS)
	$(CC) -c func.c $(OPTS)
	$(CC) main.c func.o -o main.out
clean:
	rm *.so *.o