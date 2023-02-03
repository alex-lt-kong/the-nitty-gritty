OPTS = -O2 -Wall -pedantic -Wextra 
CC=g++

main: mylibs.o
	swig -c++ -python mylibs.i
	$(CC) $(OPTS) -fPIC -c mylibs_wrap.cxx -I/usr/include/python3.9/ -IW:\\python3.9\\include
	$(CC) $(OPTS) -fPIC -shared mylibs.o mylibs_wrap.o -o _mylibs.so -LW:\\python3.9\\libs\\
	
mylibs.o: mylibs.cpp
	$(CC) $(OPTS) -fPIC -c mylibs.cpp

clean:
	rm *.o *.so
	rm mylibs_wrap.cxx