OPTS = -O3 -Wall -pedantic -Wextra 
CXX=g++

main: base-constructor.out copy-constructor.out move-constructor.out

base-constructor.out: base-constructor.cpp
	$(CXX) $(OPTS) base-constructor.cpp -o base-constructor.out
copy-constructor.out: copy-constructor.cpp
	$(CXX) $(OPTS)  copy-constructor.cpp -o copy-constructor.out
move-constructor.out: move-constructor.cpp
	$(CXX) $(OPTS) -fno-elide-constructors move-constructor.cpp -o move-constructor-noec.out
	$(CXX) $(OPTS) move-constructor.cpp -o move-constructor.out

clean:
	rm *.out