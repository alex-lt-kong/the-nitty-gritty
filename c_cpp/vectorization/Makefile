main: main.c
	gcc main.c -o main-vec-on.out -O2 -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed
	gcc main.c -o main-vec-off.out -O2 -fno-tree-vectorize

clear:
	rm *.out