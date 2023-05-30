OPTS = -O2 -Wall -pedantic -Wextra
LIBS = -levent

main: 1_hello-world.out 2_multiple-events.out

1_hello-world.out: 1_hello-world.c
	$(CC) -o 1_hello-world.out 1_hello-world.c $(OPTS) $(LIBS)

2_multiple-events.out: 2_multiple-events.c
	$(CC) -o 2_multiple-events.out 2_multiple-events.c $(OPTS) $(LIBS)

.PHONY: clean
clean:
	rm *.out
