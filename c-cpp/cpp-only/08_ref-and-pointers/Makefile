CXX = g++
CXXFLAGS = -O2 -Wall -pedantic -Wextra -fsanitize=address

all: unique_ptr.out value-ref-and-pointer-98.out value-ref-and-pointer-11.out

unique_ptr.out: unique_ptr.cpp
	$(CXX) unique_ptr.cpp -o unique_ptr.out $(CXXFLAGS)
value-ref-and-pointer-98.out: value-ref-and-pointer.cpp
	$(CXX) value-ref-and-pointer.cpp -o value-ref-and-pointer-98.out -std=c++98 $(CXXFLAGS)
value-ref-and-pointer-11.out: value-ref-and-pointer.cpp
	$(CXX) value-ref-and-pointer.cpp -o value-ref-and-pointer-11.out -std=c++11 $(CXXFLAGS)


.PHONY:
clean:
	rm *.out
