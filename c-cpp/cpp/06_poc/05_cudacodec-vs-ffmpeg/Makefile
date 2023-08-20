#CXX=/usr/local/cuda-11.8/bin/nvcc
CXX=g++
CXXFLAGS=-O2 -Wall -pedantic -Wextra
INC=-I/usr/local/include/opencv4/
LDFLAGS=-lopencv_cudacodec -lopencv_core

all: cudacodec.out

cudacodec.out: cudacodec.cpp
	$(CXX) $(INC) cudacodec.cpp -o cudacodec.out $(CXXFLAGS) $(LDFLAGS) 



.PHONY: clean
clean:
	rm *.out