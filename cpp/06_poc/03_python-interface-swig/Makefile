OPTS = -O3 -Wall -pedantic -Wextra 
CC=clang++

main: mylibs.o
	swig -c++ -python mylibs.i
	$(CC) $(OPTS) -fPIC -c mylibs_wrap.cxx -I/usr/include/python3.9/ -IW:\\python3.9\\include
	$(CC) $(OPTS) -fPIC -shared mylibs.o mylibs_wrap.o -o _mylibs.so -LW:\\python3.9\\libs\\
	
mylibs.o: mylibs.cpp
	$(CC) $(OPTS) -fPIC -c mylibs.cpp

windows:
	$(CC) $(OPTS) -c mylibs.h
	$(CC) $(OPTS) -c mylibs_wrap.cxx
	$(CC) $(OPTS) -shared mylibs.o mylibs_wrap.o -o mylibs.dll
	

clean:
	rm *.o *.so
	rm mylibs_wrap.cxx