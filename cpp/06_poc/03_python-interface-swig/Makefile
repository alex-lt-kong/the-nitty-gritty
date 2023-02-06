OPTS = -O3 -Wall -pedantic -Wextra 
CC=clang++

main: mylib.o
	swig -c++ -python -builtin mylib.i
	python3 setup.py build
	yes | cp ./build/lib.linux-x86_64-3.9/* ./python-wrapper/
windows: mylib.o
	swig -c++ -csharp -outdir ./csharp-wrapper/csharp-wrapper/swig-files/ mylib.i
	$(CC) $(OPTS) -shared mylib.o mylib_wrap.cxx -o ./csharp-wrapper/csharp-wrapper/swig-files/mylib.dll

mylib.o: mylib.cpp
	$(CC) $(OPTS) -c mylib.cpp



clean:
	rm *.o *.so *_wrap.* *.dll