# OpenCV's video manipulation backend: Cudacodec v.s. FFmpeg

* There are two ways for OpenCV to leverage Nvidia GPUs.

  1. The first way is to specifically build OpenCV that enables classes under
  the `cv::cudacodec` namespace.
      * With this approach, OpenCV is fully aware of the existence of GPU
      and the CUDA library, which hopefully translate to better optimization
      and higher performance.
      * The side effect is that we need to explicitly handle the difference
      between host (CPU/main memory) and device (GPU/vram). We also need to
      use dedicated data type such as `cv::cuda::GpuMat` to store frames
      which will otherwise be stored in `cv::Mat`.

  1. The second approach is to build FFmpeg that incorporates Nvidia Video
  Codec SDK and build OpenCV on top of FFmpeg, allowing OpenCV to leverage
  the power of GPU almost transparently.
      * Take this approach means OpenCV does not need to be aware of the
      existence of GPU/CUDA library and all existing CPU code can utilize
      GPU acceleration.
      * However, it is way less flexible as FFmpeg is mostly used only to
      encode/decode videos, if we want to manipulate frames in other ways,
      we may not be able to use GPU.

* But let's say we only need to read encoded video from a source, decode it,
applying some very simple operations (e.g., adding timestamp) and then write
to a video file, how will these two approaches compare?

## Build OpenCV

* Building an OpenCV version that supports both method proves no easy task.
Here we document them separately just for the sake of easy debugging.

### OpenCV with `cv::cudacodec` usable

* Have Nvidia GPU, driver and CUDA library properly installed (verify this
by issuing `nvidia-smi`)

* Download Nvidia's Video Codec SDK. This is a very confusing step as seems
it is properly nowhere except in a stack exchange post
[here](https://stackoverflow.com/questions/65740367/reading-a-video-on-gpu-using-c-and-cuda)
  * Long story short, we need to download Nvidia's VIdeo Codec SDK
  [here](https://developer.nvidia.com/video-codec-sdk) and copy all header files
  in `./Interface/` directory to corresponding CUDA's include directory
  (e.g., `/usr/local/cuda/targets/x86_64-linux/include/`)
  * Without this step, `OpenCV`'s `cmake` will still work, but the compiled code
  will complain:
    ```
    terminate called after throwing an instance of 'cv::Exception'
    what():  OpenCV(4.7.0) ./repos/opencv/modules/core/include/opencv2/core/private.cuda.hpp:112: error: (-213:The function/feature is not implemented) The called functionality is disabled for current build or platform in function 'throw_no_cuda'
    ```

* Prepare [opencv_contrib](https://github.com/opencv/opencv_contrib) repository.
OpenCV needs it to build `cuda` support.

* The final `cmake` command should look like below:

```
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_NVCUVID=ON \
-D WITH_NVCUVENC=ON \
-D BUILD_opencv_cudacodec=ON \
# -D CMAKE_CXX_FLAGS="-I~/repos/Video_Codec_SDK_12.1.14/Interface/" \
-D OPENCV_EXTRA_MODULES_PATH=~/repos/opencv_contrib/modules/ \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D BUILD_EXAMPLES=OFF \
..

--   NVIDIA CUDA:                   YES (ver 11.8, CUFFT CUBLAS NVCUVID NVCUVENC)
--     NVIDIA GPU arch:             35 37 50 52 60 61 70 75 80 86
--     NVIDIA PTX archs:


```

FFmpeg route:

```
apt remove libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
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
-D BUILD_SHARED_LIBS=OFF \
..
```
result from `cmake` should show 
```
ideo I/O:
--     DC1394:                      NO
--     FFMPEG:                      YES
--       avcodec:                   YES (59.37.100)
--       avformat:                  YES (59.27.100)
--       avutil:                    YES (57.28.100)
--       swscale:                   YES (6.7.100)
--       avresample:                NO
--     GStreamer:                   NO
--     v4l/v4l2:                    YES (linux/videodev2.h)

```
and versions of `avcodec`, `avformat`, etc should match those from `ffmpeg`