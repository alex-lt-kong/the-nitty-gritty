CC=/usr/local/cuda/bin/nvcc
CFLAGS=-O3
LDFLAGS=-lcublas

main: cublas.out

cublas.out: cublas.cu
	$(CC) cublas.cu -o cublas.out $(CFLAGS) $(LDFLAGS) 

.PHONY:
clean:
	rm *.out