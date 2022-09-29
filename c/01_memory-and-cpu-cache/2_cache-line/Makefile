main.out: 1st.c 2nd.c
	gcc 1st.c -o 1st-gcc.out -O3 -fno-tree-vectorize
	icc 1st.c -o 1st-icc.out -O3 -no-vec
	gcc 2nd.c -o 2nd-gcc.out -O3 -fno-tree-vectorize
	icc 2nd.c -o 2nd-icc.out -O3 -no-vec
	lscpu > ./system.env
clean:
	rm *.out