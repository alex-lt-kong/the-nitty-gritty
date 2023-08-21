#CXX=/usr/local/cuda-11.8/bin/nvcc
CXX=g++
CXXFLAGS=-O2 -Wall -pedantic -Wextra
INC=-I/usr/local/include/opencv4/
LDFLAGS=-lopencv_cudacodec -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudaarithm


all: cudacodec.out ffmpeg.out

cudacodec.out: cudacodec.cpp utils.hpp
	$(CXX) $(INC) cudacodec.cpp -o cudacodec.out $(CXXFLAGS) $(LDFLAGS) 

ffmpeg.out: ffmpeg.cpp utils.hpp
	$(CXX) $(INC) ffmpeg.cpp -o ffmpeg.out $(CXXFLAGS) $(LDFLAGS) 


.PHONY: clean
clean:
	rm *.out