OPTS = -O2 -Wall -pedantic -Wextra
INC = -I/usr/include/opencv4/
LIBS = -lopencv_highgui -lopencv_videoio -lopencv_core
CC = gcc
CXX = g++

main:
	$(CXX) 1_popen-naive.cpp -o 1_popen-naive.out $(OPTS) $(INC) $(LIBS)
	$(CXX) 2_popen-batched.cpp -o 2_popen-batched.out $(OPTS) $(INC) $(LIBS)
	

.PHONY: clean
clean:
	rm *.out