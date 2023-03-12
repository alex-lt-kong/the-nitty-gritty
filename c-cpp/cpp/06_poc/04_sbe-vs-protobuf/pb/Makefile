OPTS = -O2 -Wall -pedantic -Wextra

main: tradeData.pb.o main.cpp
	g++ main.cpp -o main.out $(OPTS) -lprotobuf tradeData.pb.o


tradeData.pb.o: ./include/tradeData.pb.cc
	g++ -c ./include/tradeData.pb.cc $(OPT) -fPIE

.PHONY: clean
	rm *.out