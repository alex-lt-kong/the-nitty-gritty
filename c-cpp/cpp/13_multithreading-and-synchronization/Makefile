CXX = g++
LIBS = -lpthread
OPTS = -O2 -Wall -pedantic -Wextra 

main: 1_hello-world.out 2_with-this.out 3_with-mutex.out 4_with-cv.out

1_hello-world.out: 1_hello-world.cpp
	$(CXX) 1_hello-world.cpp -o 1_hello-world.out $(LIBS)
2_with-this.out: 2_with-this.cpp
	$(CXX) 2_with-this.cpp -o 2_with-this.out $(LIBS)
3_with-mutex.out: 3_with-mutex.cpp
	$(CXX) 3_with-mutex.cpp -o 3_with-mutex.out $(LIBS)
4_with-cv.out: 4_with-cv.cpp
	$(CXX) 4_with-cv.cpp -o 4_with-cv.out $(LIBS)

.PHONY: clean
clean:
	rm *.out