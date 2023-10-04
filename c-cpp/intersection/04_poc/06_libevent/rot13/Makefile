CC=gcc
OPTS = -O2 -Wall -pedantic -Wextra -Wc++-compat

main: 1_fork.out 2_select.out

1_fork.out: 1_fork.c
	$(CC) 1_fork.c -o 1_fork.out $(OPTS)

2_select.out: 2_select.c
	$(CC) 2_select.c -o 2_select.out $(OPTS)

.PHONY:
clean:
	rm *.out
