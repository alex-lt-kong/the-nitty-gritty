OPTS = -O3 -Wall -pedantic -Wc++-compat -gdwarf-4 -Wextra -Wwrite-strings -Wno-parentheses -Wpedantic -Warray-bounds -Wmissing-braces -Wconversion  -Wstrict-prototypes -fanalyzer

main:
	gcc main-naive.c -o main-naive.out $(OPTS)
	gcc main-poll.c -o main-poll.out $(OPTS)
	gcc sub.c -o sub.out $(OPTS)