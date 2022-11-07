OPT = -O3 -Wall -g

main: 1_hello-world.out 2_dynamic-input.out
	g++ --version > gpp.version

1_hello-world.out: 1_hello-world.cpp
	g++ -o 1_hello-world.out 1_hello-world.cpp $(OPT)
	objdump --disassembler-options "intel" --disassemble=main --source 1_hello-world.out > 1_hello-world.asm

2_dynamic-input.out: 2_dynamic-input.cpp
	g++ -o 2_dynamic-input.out 2_dynamic-input.cpp $(OPT)
	objdump --disassembler-options "intel" --demangle --source 2_dynamic-input.out > 2_dynamic-input.asm


clean:
	rm *.out