OPT = -O2 -Wall -pedantic -Wextra -g -Og
CXX = g++

main: 1_hello-world.out 2_dynamic-input.out 3_noinline.out 4_edge-cases.out 5_trait.out
	$(CXX) --version > gpp.version

1_hello-world.out: 1_hello-world.cpp
	$(CXX) -o 1_hello-world.out 1_hello-world.cpp $(OPT)
	objdump --disassembler-options "intel" --disassemble=main --demangle --source 1_hello-world.out > 1_hello-world.asm

2_dynamic-input.out: 2_dynamic-input.cpp
	$(CXX) -o 2_dynamic-input.out 2_dynamic-input.cpp $(OPT)
	objdump --disassembler-options "intel" --demangle --source 2_dynamic-input.out > 2_dynamic-input.asm

3_noinline.out: 3_noinline.cpp
	$(CXX) -o 3_noinline.out 3_noinline.cpp $(OPT)
	objdump --disassembler-options "intel" --demangle --source 3_noinline.out > 3_noinline.asm
	
4_edge-cases.out: 4_edge-cases.cpp
	$(CXX) -o 4_edge-cases.out 4_edge-cases.cpp $(OPT)

5_trait.out: 5_trait.cpp
	$(CXX) -o 5_trait.out 5_trait.cpp $(OPT)

clean:
	rm *.out