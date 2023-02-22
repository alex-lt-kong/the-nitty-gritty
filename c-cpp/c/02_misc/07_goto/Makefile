OPTS = -O3 -Wall -pedantic -Wc++-compat 

main:
	gcc bypasses-initialization.c -o bypasses-initialization.out $(OPTS)
	g++ bypasses-initialization.cpp -o bypasses-initialization-cpp.out $(OPTS)
	gcc bypasses-declaration.c -o bypasses-declaration.out $(OPTS)
	gcc bypasses-declaration-vla.c -o bypasses-declaration-vla.out $(OPTS)

clean:
	rm *.out