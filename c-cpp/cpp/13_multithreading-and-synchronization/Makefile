CXX = g++
LIBS = -lpthread
OPTS = -O2 -Wall -pedantic -Wextra 

main: 1_hello-world.out 2_with-this.out

1_hello-world.out: 1_hello-world.cpp
	$(CXX) 1_hello-world.cpp -o 1_hello-world.out $(LIBS)
2_with-this.out: 2_with-this.cpp
	$(CXX) 2_with-this.cpp -o 2_with-this.out $(LIBS)

.PHONY: clean
clean:
	rm *.out