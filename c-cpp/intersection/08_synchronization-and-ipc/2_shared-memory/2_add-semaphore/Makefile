OPTS = -O2 -Wall -Wc++-compat -pedantic -Wextra
LIBS = -lrt -lpthread
CC = clang

main:
	$(CC) -o shm-writer.out shm-writer.c $(OPTS) $(LIBS)
	$(CC) -o shm-reader.out shm-reader.c $(OPTS) $(LIBS)

.PHONY: clean
clean:
	rm *.out