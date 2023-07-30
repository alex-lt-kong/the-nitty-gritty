NVCC=/usr/local/cuda/bin/nvcc
CXX=g++
NVCFLAGS=-O3
NVLDFLAGS=-lcublas
OPCFLAGS=-O3 -Wall -pedantic -Wextra -Wc++-compat
OPLDFLAGS=-lopenblas

main: cublas.bin openblas.bin

cublas.bin: cublas.cu ../../utils.hpp ../../utils.h
	$(NVCC) cublas.cu -o cublas.bin $(NVCFLAGS) $(NVLDFLAGS) 
openblas.bin: openblas.cpp ../../utils.hpp ../../utils.h
	$(CXX) openblas.cpp -o openblas.bin $(OPCFLAGS) $(OPLDFLAGS) 

.PHONY:
clean:
	rm *.bin