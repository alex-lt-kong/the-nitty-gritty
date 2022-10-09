OPT = -Wall -O3

main: 1st.out 2nd.out 3rd.out lscpu.txt

1st.out: 1st.c
	gcc -o 1st.out 1st.c $(OPT)

2nd.out: 2nd.c
	gcc -o 2nd.out 2nd.c $(OPT) -lm

3rd.out: 3rd.c
	gcc -o 3rd.out 3rd.c $(OPT) -lm

lscpu.txt:
	lscpu > ./lscpu.txt