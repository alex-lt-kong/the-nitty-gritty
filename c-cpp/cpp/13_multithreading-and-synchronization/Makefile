CXX = g++
LIBS = -lpthread
OPTS = -O2 -Wall -pedantic -Wextra 

main: 1_hello-world.out 2_join-and-detach.out 3_with-this.out 4_with-mutex.out 5_condition-var.out

1_hello-world.out: 1_hello-world.cpp
	$(CXX) 1_hello-world.cpp -o 1_hello-world.out $(LIBS)
2_join-and-detach.out: 2_join-and-detach.cpp
	$(CXX) 2_join-and-detach.cpp -o 2_join-and-detach.out $(LIBS)
3_with-this.out: 3_with-this.cpp common.h
	$(CXX) 3_with-this.cpp -o 3_with-this.out $(LIBS)
4_with-mutex.out: 4_with-mutex.cpp common.h
	$(CXX) 4_with-mutex.cpp -o 4_with-mutex.out $(LIBS)
5_condition-var.out: 5_condition-var.cpp common.h
	$(CXX) 5_condition-var.cpp -o 5_condition-var.out $(LIBS)

.PHONY: clean
clean:
	rm *.out