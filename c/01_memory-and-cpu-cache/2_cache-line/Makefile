main.out: 1st.c
	gcc 1st.c -o 1st-gcc.out -O3 -Wall
	icc 1st.c -o 1st-icc.out -O3 -Wall
	gcc 2nd.c -o 2nd-gcc.out -O3 -Wall
	icc 2nd.c -o 2nd-icc.out -O3 -Wall
	lscpu > ./system.env
clean:
	rm *.out