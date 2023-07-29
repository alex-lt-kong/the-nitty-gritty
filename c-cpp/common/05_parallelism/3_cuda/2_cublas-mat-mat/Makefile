CC=/usr/local/cuda/bin/nvcc
CFLAGS=-O3
LDFLAGS=-lcublas

main:
	$(CC) cublas_gemm.cu -o cublas_gemm.out $(CFLAGS) $(LDFLAGS) 

.PHONY:
clean:
	rm *.out