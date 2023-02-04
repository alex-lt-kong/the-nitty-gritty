OPTS = -O3 -Wall -pedantic -Wextra 
CC=clang++

main: mylib.o
	swig -c++ -python mylib.i
	python3 setup.py build
	yes | cp ./build/lib.linux-x86_64-3.9/* ./python-wrapper/

mylib.o: mylib.cpp
	$(CC) $(OPTS) -fPIC -c mylib.cpp

clean:
	rm *.o *.so
	rm *_wrap.*