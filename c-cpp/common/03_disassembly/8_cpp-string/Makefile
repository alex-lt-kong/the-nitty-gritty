main.out: main.cpp
	g++ main.cpp -o main.out -O1 -g
#objdump --disassembler-options "intel" -S ./main.out > ./main.asm
	objdump --disassembler-options "intel" --demangle --disassemble=main --source main.out > ./main.asm

.PHONY: clean
clean:
	rm *.out