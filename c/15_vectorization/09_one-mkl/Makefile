COMMON_OPT = -Wall -O3
MKL_ROOT = /opt/intel/oneapi/mkl/2022.1.0
main:
	gcc -I$(MKL_ROOT)/include/ main.c -L $(MKL_ROOT)/lib/64 -L $(MKL_ROOT)/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -lpthread -lm -O3 -o main.out