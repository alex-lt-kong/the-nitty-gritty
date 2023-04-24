CC = gcc
LIBS = -lpthread -lglib-2.0
OPTS = -O2 -Wall -pedantic -Wextra -Wc++-compat
INC = -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/

main: 1_hello-world.out 2_join-and-detach.out 3_with-parameters.out 4_multiple-threads.out 5_with-mutex.out 6_glibc-queue.out

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
6_glibc-queue.out: 6_glibc-queue.c common.h
	$(CC) 6_glibc-queue.c -o 6_glibc-queue.out $(LIBS) $(INC)

.PHONY: clean
clean:
	rm *.out