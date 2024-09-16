OPTS = -Wall -pedantic -Wextra -g -Og
CXX = g++

main: main.cpp
	$(CXX) $(OPTS) main.cpp -o main.out	

.PHONY: clean
clean:
	rm *.out *.o *.asm
disassemble:
	objdump --disassembler-options "intel" --no-show-raw-insn --demangle --disassemble=main --source ./main.out > ./main.asm