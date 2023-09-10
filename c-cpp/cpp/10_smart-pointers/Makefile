CXX=g++
CXXFLAGS=-O3 -Wall -pedantic -Wextra

all: unique_ptr.out

unique_ptr.out: unique_ptr.cpp
	$(CXX) unique_ptr.cpp -o unique_ptr.out $(CXXFLAGS)

.PHONY:
clean:
	rm *.out
