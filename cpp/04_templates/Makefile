OPT = -O3 -Wall -g

main: 1_hello-world.out

1_hello-world.out: 1_hello-world.cpp
	g++ -o 1_hello-world.out 1_hello-world.cpp $(OPT)
	objdump --disassembler-options "intel" --disassemble=main --source 1_hello-world.out > 1_hello-world.asm

clean:
	rm *.out