OPTS = -O3 -Wall -pedantic -Wc++-compat -Wextra 
CC=clang++

main: libs.o
	swig -c++ -python libs.i
	$(CC) $(OPTS) -fPIC -c libs_wrap.cxx -I/usr/include/python3.9/
	$(CC) $(OPTS) -shared libs.o libs_wrap.o -o _libs.so

libs.o: libs.cpp
	$(CC) $(OPTS) -fPIC -c libs.cpp

clean:
	rm *.o *.so
	rm libs_wrap.cxx