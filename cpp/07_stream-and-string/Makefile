OPT = -O3 -Wall -g

main: 1_stringstream.out 2_string-to-unix-ts.out

1_stringstream.out: 1_stringstream.cpp
	g++ -o 1_stringstream.out 1_stringstream.cpp $(OPT)

2_string-to-unix-ts.out: 2_string-to-unix-ts.cpp
	g++ -o 2_string-to-unix-ts.out 2_string-to-unix-ts.cpp $(OPT)

clean:
	rm *.out