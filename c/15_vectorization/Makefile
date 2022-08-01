main: main.c
	gcc main.c -o main-vec-on.out -O3 -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed
	gcc main.c -o main-vec-off.out -O3 -fno-tree-vectorize

clear:
	rm *.out