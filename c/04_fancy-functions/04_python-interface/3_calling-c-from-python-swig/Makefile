OPTS = -O3 -Wall -pedantic -Wc++-compat -Wextra 
CC=clang

main: libs.o
	/usr/bin/swig -python libs.i
	$(CC) $(OPTS) -fPIC -c libs_wrap.c -I/usr/include/python3.9/
	$(CC) $(OPTS) -shared libs.o libs_wrap.o -o _libs.so

libs.o: libs.c
	$(CC) $(OPTS)  -fPIC -c libs.c

clean:
	rm *.o *.so
	rm libs_wrap.c