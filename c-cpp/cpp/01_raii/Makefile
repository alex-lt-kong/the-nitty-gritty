CXX=g++
CXXFLAGS=-O3 -Wall -pedantic -Wextra

all: main.out pointer.out

main.out: main.cpp
	$(CXX) main.cpp -o main.out $(CXXFLAGS)

pointer.out: pointer.cpp
	$(CXX) pointer.cpp -o pointer.out $(CXXFLAGS)

.PHONY:
clean:
	rm *.out