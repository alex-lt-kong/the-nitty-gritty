OPT = -O3 -Wall -g

main: 1_hello-world.out 2_dynamic-input.out 3_noinline.out 4_edge-cases.out 5_test.out
	g++ --version > gpp.version

1_hello-world.out: 1_hello-world.cpp
	g++ -o 1_hello-world.out 1_hello-world.cpp $(OPT)
	objdump --disassembler-options "intel" --disassemble=main --demangle --source 1_hello-world.out > 1_hello-world.asm

2_dynamic-input.out: 2_dynamic-input.cpp
	g++ -o 2_dynamic-input.out 2_dynamic-input.cpp $(OPT)
	objdump --disassembler-options "intel" --demangle --source 2_dynamic-input.out > 2_dynamic-input.asm

3_noinline.out: 3_noinline.cpp
	g++ -o 3_noinline.out 3_noinline.cpp $(OPT)
	objdump --disassembler-options "intel" --demangle --source 3_noinline.out > 3_noinline.asm
	
4_edge-cases.out: 4_edge-cases.cpp
	g++ -o 4_edge-cases.out 4_edge-cases.cpp $(OPT)

5_test.out: 5_test.cpp
	g++ -o 5_test.out 5_test.cpp $(OPT)

clean:
	rm *.out