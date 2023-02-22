OPTS = -O3 -Wall -pedantic -Wextra 
CXX=g++

main: base-constructor.out copy-constructor.out move-constructor.out copy-elision.out

base-constructor.out: base-constructor.cpp
	$(CXX) $(OPTS) base-constructor.cpp -o base-constructor.out
copy-constructor.out: copy-constructor.cpp
	$(CXX) $(OPTS)  copy-constructor.cpp -o copy-constructor.out
move-constructor.out: move-constructor.cpp
	$(CXX) $(OPTS) -fno-elide-constructors move-constructor.cpp -o move-constructor-noec.out
	$(CXX) $(OPTS) move-constructor.cpp -o move-constructor.out
copy-elision.out:  copy-elision.cpp
	$(CXX) $(OPTS) copy-elision.cpp -g -o copy-elision.out
# gdb --quiet --eval-command="set disassembly-flavor intel" --eval-command="disassemble /m getStringVec" --batch ./copy-elision.out > ./copy-elision.asm

clean:
	rm *.out