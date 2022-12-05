main: func_c.c main.c
	gcc -fPIC -shared -O3 -o func_c.so func_c.c
	gcc main.c -O3 -o main.out
	#-Ofast is even more aggressive than -O3
	python3 ./main.py
	./main.out
