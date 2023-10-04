OPTS = -O2 -Wall -pedantic -Wextra
INC = -I/usr/include/opencv4/
LIBS = -lopencv_highgui -lopencv_videoio -lopencv_core
CC = gcc
CXX = g++

main: 1_popen-naive.out 2_popen-batched.out 3_with-signal-handler.out

1_popen-naive.out: 1_popen-naive.cpp common.h
	$(CXX) 1_popen-naive.cpp -o 1_popen-naive.out $(OPTS) $(INC) $(LIBS)
2_popen-batched.out: 2_popen-batched.cpp common.h
	$(CXX) 2_popen-batched.cpp -o 2_popen-batched.out $(OPTS) $(INC) $(LIBS)
3_with-signal-handler.out: 3_with-signal-handler.cpp common.h
	$(CXX) 3_with-signal-handler.cpp -o 3_with-signal-handler.out $(OPTS) $(INC) $(LIBS) -lrt -lpthread
	

.PHONY: clean
clean:
	rm *.out