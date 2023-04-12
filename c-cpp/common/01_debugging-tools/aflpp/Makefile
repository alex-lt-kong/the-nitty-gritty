main: main.c
	afl-clang main.c -o main-c.out

.PHONY: test clean
test:
	afl-fuzz -i input/ -o output/ ./main-c.out
clean:
	rm *.out
