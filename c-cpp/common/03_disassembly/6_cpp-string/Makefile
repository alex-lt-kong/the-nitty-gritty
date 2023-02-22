main.out: main.cpp
	g++ main.cpp -o main.out -O2
#objdump --disassembler-options "intel" -S ./main.out > ./main.asm
	objdump --disassembler-options "intel" --disassemble=main -S main.out > ./main.asm
.PHONY: clean
clean:
	rm *.out