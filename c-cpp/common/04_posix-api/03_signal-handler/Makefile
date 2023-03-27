OPTS = -O2 -Wall -pedantic -Wc++-compat
CC = gcc

main:
	$(CC) 1_hello-world.c -o 1_hello-world.out $(OPTS)
	$(CC) 2_multithreading.c -o 2_multithreading.out $(OPTS) -lrt -lpthread
	

.PHONY: clean
clean:
	rm *.out