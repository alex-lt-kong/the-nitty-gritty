CXX = g++
CXXFLAGS = -O2 -Wall -pedantic -Wextra -fsanitize=address

all: unique_ptr.out main-98.out main-11.out

unique_ptr.out: unique_ptr.cpp
	$(CXX) unique_ptr.cpp -o unique_ptr.out $(CXXFLAGS)
main-98.out: value-ref-and-pointer.cpp
	$(CXX) value-ref-and-pointer.cpp -o main-98.out -std=c++98 $(CXXFLAGS)
main-11.out: value-ref-and-pointer.cpp
	$(CXX) value-ref-and-pointer.cpp -o main-11.out -std=c++11 $(CXXFLAGS)


.PHONY:
clean:
	rm *.out
