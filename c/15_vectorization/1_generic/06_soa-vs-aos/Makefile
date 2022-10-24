main: main-aos.c main-soa.c func.o utils.o
	icc main-aos.c func-icc-on.o utils.o -o main-icc-aos-on.out -g -O3 -lm
	icc main-aos.c func-icc-no.o utils.o -o main-icc-aos-no.out -g -O3 -lm -no-vec
	icc main-soa.c func-icc-on.o utils.o -o main-icc-soa-on.out -g -O3 -lm
	icc main-soa.c func-icc-no.o utils.o -o main-icc-soa-no.out -g -O3 -lm -no-vec
	gcc main-aos.c func-gcc-on.o utils.o -o main-gcc-aos-on.out -g -O3 -lm
	gcc main-aos.c func-gcc-no.o utils.o -o main-gcc-aos-no.out -g -O3 -lm -fno-tree-vectorize 
	gcc main-soa.c func-gcc-on.o utils.o -o main-gcc-soa-on.out -g -O3 -lm
	gcc main-soa.c func-gcc-no.o utils.o -o main-gcc-soa-no.out -g -O3 -lm -fno-tree-vectorize 

utils.o: ../utils.c ../utils.h
	icc -fpic -c ../utils.c -O3

func.o: func.c func.h
	icc -c func.c -o func-icc-on.o -g -O2 		  -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=./func-icc-on.report
	gdb -batch -ex 'file ./func-icc-on.o' -ex 'disassemble /m floating_division_aos' -ex 'disassemble /m floating_division_soa' > ./func-icc-on.asm

	icc -c func.c -o func-icc-no.o -g -O3 -no-vec -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=./func-icc-no.report
	gdb -batch -ex 'file ./func-icc-no.o' -ex 'disassemble /m floating_division_aos' -ex 'disassemble /m floating_division_soa' > ./func-icc-no.asm

	gcc -c func.c -o func-gcc-on.o -g -O3    -ftree-vectorize -fopt-info-vec-all
	gdb -batch -ex 'file ./func-gcc-on.o' -ex 'disassemble /m floating_division_aos' -ex 'disassemble /m floating_division_soa' > ./func-gcc-on.asm

	gcc -c func.c -o func-gcc-no.o -g -O3 -fno-tree-vectorize -fopt-info-vec-all
	gdb -batch -ex 'file ./func-gcc-no.o' -ex 'disassemble /m floating_division_aos' -ex 'disassemble /m floating_division_soa' > ./func-gcc-no.asm

clear:
	rm *.out *.o
