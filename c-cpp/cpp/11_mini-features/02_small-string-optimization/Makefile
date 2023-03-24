OPTS = -O2 -Wall -pedantic -Wextra -g -Og
CXX = g++

main: main.o string-demo.o
	$(CXX) $(OPTS) main.o string-demo.o -o main.out	

main.o: main.cpp
	$(CXX) $(OPTS) -fPIC -c main.cpp
string-demo.o: string-demo.cpp string-demo.h
	$(CXX) $(OPTS) -fPIC -c string-demo.cpp


.PHONY: clean
clean:
	rm *.out *.o *.asm
disassemble:
	objdump --disassembler-options "intel" --no-show-raw-insn --demangle --source ./main.out > ./main.asm