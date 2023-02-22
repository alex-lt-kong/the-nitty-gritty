main.out: main.c
	gcc main.c -o main.out
	objdump --disassembler-options "intel" -S ./main.out > ./main.asm
	
clean:
	rm *.out