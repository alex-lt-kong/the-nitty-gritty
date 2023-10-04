CXX=clang++
CXXFLAGS=-O2 -Wall -pedantic -Wextra 

all: main.out

main.out: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main.out

.PHONY:
clean:
	rm *.out