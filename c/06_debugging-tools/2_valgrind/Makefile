main: main.c
# -ggdb3 is even more detailed than -g, this helps valgrind to track the
# origin of some issues
	gcc -o main.out main.c -O1 -Wall -ggdb3
	valgrind valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./main.out
	