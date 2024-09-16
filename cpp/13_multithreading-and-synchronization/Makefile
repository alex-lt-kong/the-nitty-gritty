CXX = g++
LDFLAGS = -lpthread
CXXFLAGS = -O2 -Wall -pedantic -Wextra 

main: all

1_hello-world.out: 1_hello-world.cpp
	$(CXX) 1_hello-world.cpp -o 1_hello-world.out $(LDFLAGS) $(CXXFLAGS)
2_join-and-detach.out: 2_join-and-detach.cpp
	$(CXX) 2_join-and-detach.cpp -o 2_join-and-detach.out $(LDFLAGS) $(CXXFLAGS)
3_with-this.out: 3_with-this.cpp common.h
	$(CXX) 3_with-this.cpp -o 3_with-this.out $(LDFLAGS) $(CXXFLAGS)
4_with-mutex.out: 4_with-mutex.cpp common.h
	$(CXX) 4_with-mutex.cpp -o 4_with-mutex.out $(LDFLAGS) $(CXXFLAGS)
5_condition-var.out: 5_condition-var.cpp common.h
	$(CXX) 5_condition-var.cpp -o 5_condition-var.out $(LDFLAGS) $(CXXFLAGS)

.PHONY: clean
clean:
	rm *.out