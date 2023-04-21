CC = gcc
LIBS = -lpthread
OPTS = -O2 -Wall -pedantic -Wextra -Wc++-compat

main: 1_hello-world.out 2_join-and-detach.out 3_with-parameters.out 4_multiple-threads.out 5_with-mutex.out

1_hello-world.out: 1_hello-world.c
	$(CC) 1_hello-world.c -o 1_hello-world.out $(LIBS)
2_join-and-detach.out: 2_join-and-detach.c
	$(CC) 2_join-and-detach.c -o 2_join-and-detach.out $(LIBS)
3_with-parameters.out: 3_with-parameters.c common.h
	$(CC) 3_with-parameters.c -o 3_with-parameters.out $(LIBS)
4_multiple-threads.out: 4_multiple-threads.c common.h
	$(CC) 4_multiple-threads.c -o 4_multiple-threads.out $(LIBS)
5_with-mutex.out: 5_with-mutex.c common.h
	$(CC) 5_with-mutex.c -o 5_with-mutex.out $(LIBS)
#5_condition-var.out: 5_condition-var.cpp common.h
#	$(CXX) 5_condition-var.cpp -o 5_condition-var.out $(LIBS)

.PHONY: clean
clean:
	rm *.out