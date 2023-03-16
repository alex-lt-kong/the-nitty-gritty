OPTS = -O2 -Wall -Wc++-compat -pedantic -Wextra -g
CC = gcc

main: main.o factorial.o
	$(CC) $(OPTS) main.o factorial.o -o main.out	

main.o: main.c factorial.o
	$(CC) $(OPTS) -fPIC -c main.c
string-demo.o: factorial.c string-demo.h
	$(CC) $(OPTS) -fPIC -c factorial.c


.PHONY: clean
clean:
	rm *.out *.o *.asm
disassemble:
	objdump --disassembler-options "intel" --no-show-raw-insn --source ./factorial.o > ./factorial.asm
