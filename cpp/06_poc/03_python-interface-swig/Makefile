OPTS = -O3 -Wall -pedantic -Wc++-compat -Wextra 
CC=g++

main: mylibs.o
	swig -c++ -python mylibs.i
	$(CC) $(OPTS) -c mylibs_wrap.cxx -I/usr/include/python3.9/ -IW:\\python3.11\\include
	$(CC) $(OPTS) -shared mylibs.o mylibs_wrap.o -o _mylibs.so -LW:\\python3.11\\libs\\  -lpython311

mylibs.o: mylibs.cpp
	$(CC) $(OPTS)   -c mylibs.cpp

clean:
	rm *.o *.so
	rm mylibs_wrap.cxx