COMMON_OPT = -Wall -O3 -Ofast
MKL_ROOT = /opt/intel/oneapi/mkl/latest

main:
	icc -I$(MKL_ROOT)/include/ func.c -L $(MKL_ROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -fPIC -shared -o func.so $(COMMON_OPT)
	icc -I$(MKL_ROOT)/include/ func.c -L $(MKL_ROOT)/lib/intel64 -qmkl=parallel -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -o func.out $(COMMON_OPT)