CC = gcc
LDFLAGS = -lpthread -lglib-2.0
CXXFLAGS = -O2 -Wall -pedantic -Wextra -Wc++-compat
INC = -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/

main: all

1_hello-world.out: 1_hello-world.c
	$(CC) 1_hello-world.c -o 1_hello-world.out $(LDFLAGS) $(CXXFLAGS)
2_join-and-detach.out: 2_join-and-detach.c
	$(CC) 2_join-and-detach.c -o 2_join-and-detach.out $(LDFLAGS) $(CXXFLAGS)
3_with-parameters.out: 3_with-parameters.c common.h
	$(CC) 3_with-parameters.c -o 3_with-parameters.out $(LDFLAGS) $(CXXFLAGS)
4_multiple-threads.out: 4_multiple-threads.c common.h
	$(CC) 4_multiple-threads.c -o 4_multiple-threads.out $(LDFLAGS) $(CXXFLAGS)
5_with-mutex.out: 5_with-mutex.c common.h
	$(CC) 5_with-mutex.c -o 5_with-mutex.out $(LDFLAGS) $(CXXFLAGS)
6_glibc-queue.out: 6_glibc-queue.c common.h
	$(CC) 6_glibc-queue.c -o 6_glibc-queue.out $(LDFLAGS) $(INC) $(CXXFLAGS)
7_condition-var.out: 7_condition-var.c common.h
	$(CC) 7_condition-var.c -o 7_condition-var.out $(LDFLAGS) $(INC) $(CXXFLAGS)

.PHONY: clean
clean:
	rm *.out