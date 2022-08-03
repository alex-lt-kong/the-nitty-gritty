main: main.c func.o utils.o	
	icc main.c func.o utils.o -o main-icc-on.out -g -O3 -lm 

utils.o: ../utils.c ../utils.h
	icc -c ../utils.c -O3

func.o: func.c func.h
	icc -c func.c -O3 -qopt-report:1 -qopt-report-phase:vec

clear:
	rm *.out *.o