OPTS = -O2 -Wall -pedantic -Wextra -g
CC = gcc

main: main.o vla.o
	$(CC) $(OPTS) main.o vla.o -o main.out	

main.o: main.c vla.o
	$(CC) $(OPTS) -fPIC -c main.c
string-demo.o: vla.c string-demo.h
	$(CC) $(OPTS) -fPIC -c vla.c


.PHONY: clean
clean:
	rm *.out *.o *.asm
disassemble:
	objdump --disassembler-options "intel" --no-show-raw-insn --source ./vla.o > ./vla.asm