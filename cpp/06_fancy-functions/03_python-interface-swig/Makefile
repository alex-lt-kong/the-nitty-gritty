OPTS = -O3 -Wall -pedantic -Wc++-compat -Wextra 
CC=clang++

main: mylibs.o
	swig -c++ -python mylibs.i
	$(CC) $(OPTS) -fPIC -c mylibs_wrap.cxx -I/usr/include/python3.9/
	$(CC) $(OPTS) -shared mylibs.o mylibs_wrap.o -o _mylibs.so

mylibs.o: mylibs.cpp
	$(CC) $(OPTS) -fPIC -c mylibs.cpp

clean:
	rm *.o *.so
	rm mylibs_wrap.cxx