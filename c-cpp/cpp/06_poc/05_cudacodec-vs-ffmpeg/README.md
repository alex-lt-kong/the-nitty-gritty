```
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.7.0) /home/mamsds/repos/opencv/modules/core/include/opencv2/core/private.cuda.hpp:112: error: (-213:The function/feature is not implemented) The called functionality is disabled for current build or platform in function 'throw_no_cuda'


cp /home/mamsds/repos/Video_Codec_SDK_12.1.14/Interface/* /usr/local/cuda-11.8/targets/x86_64-linux/include/

cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_NVCUVID=ON \
-D WITH_NVCUVENC=ON \
-D BUILD_opencv_cudacodec=ON \
-D CMAKE_CXX_FLAGS="-I~/repos/Video_Codec_SDK_12.1.14/Interface/" \
-D OPENCV_EXTRA_MODULES_PATH=~/repos/opencv_contrib/modules/ \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D BUILD_EXAMPLES=OFF \
..

--   NVIDIA CUDA:                   YES (ver 11.8, CUFFT CUBLAS NVCUVID NVCUVENC)
--     NVIDIA GPU arch:             35 37 50 52 60 61 70 75 80 86
--     NVIDIA PTX archs:


```