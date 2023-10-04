CXX=g++
CXXFLAGS=-O3 -Wall -pedantic -Wextra

all: 01_naive-example.out 02_raw-pointers-with-raii.out

01_naive-example.out: 01_naive-example.cpp
	$(CXX) 01_naive-example.cpp -o 01_naive-example.out $(CXXFLAGS)

02_raw-pointers-with-raii.out: 02_raw-pointers-with-raii.cpp
	$(CXX) 02_raw-pointers-with-raii.cpp -o 02_raw-pointers-with-raii.out $(CXXFLAGS)

.PHONY:
clean:
	rm *.out