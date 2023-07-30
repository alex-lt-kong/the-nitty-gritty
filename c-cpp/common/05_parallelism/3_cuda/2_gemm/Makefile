CC=/usr/local/cuda/bin/nvcc
CFLAGS=-O3
LDFLAGS=-lcublas

main: cublas.bin

cublas.bin: cublas.cu ../../utils.hpp
	$(CC) cublas.cu -o cublas.bin $(CFLAGS) $(LDFLAGS) 

.PHONY:
clean:
	rm *.bin