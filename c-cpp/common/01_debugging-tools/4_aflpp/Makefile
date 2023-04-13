#export AFL_USE_ASAN=1
#export AFL_USE_MSAN=1
export AFL_USE_UBSAN=1
export AFL_LLVM_LAF_ALL=make1

CC=afl-clang-fast
OPTS=-O2

main: main-buggy.out main-fixed.out

main-buggy.out: main-buggy.c
	$(CC) -v main-buggy.c -o main-buggy.out $(OPTS)
main-fixed.out: main-fixed.c
	$(CC) -v main-fixed.c -o main-fixed.out $(OPTS)

.PHONY: test clean
test-buggy:
	afl-fuzz -i input/ -o output/ ./main-buggy.out
test-fixed:
	afl-fuzz -i input/ -o output/ ./main-fixed.out
clean:
	rm *.out
