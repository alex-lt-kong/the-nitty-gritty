CC = gcc
OPTS = -O2 -Wall -pedantic -Wextra -Wc++-compat -Wconversion


main: 1_hello-world.out 2_quirks.out

1_hello-world.out: 1_hello-world.c
	$(CC) 1_hello-world.c -o 1_hello-world.out $(OPTS)
2_quirks.out: 2_quirks.c
	$(CC) 2_quirks.c -o 2_quirks.out $(OPTS)

.PHONY: clean
clean:
	rm *.out
