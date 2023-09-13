CXX=clang++
CXXFLAGS=-O2 -Wall -pedantic -Wextra 


main: base-constructor.out copy-constructor.out move-constructor.out copy-elision.out lvalue-vs-rvalue.out

lvalue-vs-rvalue.out:  lvalue-vs-rvalue.cpp
	$(CXX) $(CXXFLAGS) lvalue-vs-rvalue.cpp -g -o lvalue-vs-rvalue.out
base-constructor.out: base-constructor.cpp
	$(CXX) $(CXXFLAGS) base-constructor.cpp -o base-constructor.out
copy-constructor.out: copy-constructor.cpp
	$(CXX) $(CXXFLAGS)  copy-constructor.cpp -o copy-constructor.out
move-constructor.out: move-constructor.cpp
	$(CXX) $(CXXFLAGS) -fno-elide-constructors move-constructor.cpp -o move-constructor-noec.out
	$(CXX) $(CXXFLAGS) move-constructor.cpp -o move-constructor.out
copy-elision.out:  copy-elision.cpp
	$(CXX) $(CXXFLAGS) copy-elision.cpp -g -o copy-elision.out
# gdb --quiet --eval-command="set disassembly-flavor intel" --eval-command="disassemble /m getStringVec" --batch ./copy-elision.out > ./copy-elision.asm

.PHONY:
clean:
	rm *.out