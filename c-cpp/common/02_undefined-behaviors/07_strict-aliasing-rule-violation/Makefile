OPTS = -O2 -Wall -g

main: 
	gcc main.c -o main.out ${OPTS} 
	gcc lib.c -fPIC -c ${OPTS}

.PHONY: clean disassemble
disassemble:
	objdump --disassembler-options "intel"  --no-show-raw-insn --source ./lib.o > ./lib.asm
clean:
	rm *.out *.o