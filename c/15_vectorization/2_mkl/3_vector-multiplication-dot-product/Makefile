COMMON_OPT = -Wall -O3 -Ofast
MKL_ROOT = /opt/intel/oneapi/mkl/latest

main:
#	icc -I$(MKL_ROOT)/include/ func.c -L $(MKL_ROOT)/lib/intel64 -qmkl=parallel -lmkl_intel_lp64 -lmkl_core -lpthread -lm -fPIC -shared -o func.so $(COMMON_OPT)
	icc -I$(MKL_ROOT)/include/ func.c -DMKL_ILP64  -I"${MKL_ROOT}/include" -L $(MKL_ROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -fPIC -shared -o func.so $(COMMON_OPT) -qopt-report-file=./func.so.report -g
 