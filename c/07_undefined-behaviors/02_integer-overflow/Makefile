OPTS =  -Wall

main:
	gcc main-int.c -o main-int-o1.out -O1 ${OPTS}
	gcc main-int.c -o main-int-o2.out -O2 ${OPTS}
	gcc main-int.c -o main-int-o3.out -O3 ${OPTS}
	gcc main-uint.c -o main-uint.out -O3 ${OPTS}
clean:
	rm *.out