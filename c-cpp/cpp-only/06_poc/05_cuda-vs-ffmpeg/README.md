# OpenCV's video manipulation backend: Cudacodec v.s. FFmpeg

* There are two ways for OpenCV to leverage Nvidia GPUs.

  1. The first way is to specifically build OpenCV that enables classes under
  the `cv::cudacodec` namespace (e.g., `cv::cudacodec::VideoReader()` and
  `cv::cudacodec::VideoWriter()`).
      * With this approach, OpenCV is fully aware of the existence of GPU
      and the CUDA library, which hopefully translates to better optimization
      and higher performance.
      * The side effect is that we need to explicitly handle the difference
      between host (CPU/main memory) and device (GPU/vram). We also need to
      use dedicated data type such as `cv::cuda::GpuMat` to store frames
      which will otherwise be stored in `cv::Mat`.

  1. The second approach is to build FFmpeg that incorporates
  [Nvidia Video Codec SDK](https://developer.nvidia.com/video-codec-sdk)
  and build OpenCV on top of FFmpeg, allowing OpenCV to leverage
  the power of GPU almost transparently.
      * Takinh this approach means OpenCV does not need to be aware of the
      existence of GPU/CUDA library and all existing CPU code can utilize
      GPU acceleration.
      * However, it is way less flexible as FFmpeg is mostly used only to
      encode/decode videos, if we want to manipulate frames in other ways,
      we may not be able to use GPU.

* But let's say we only need to read encoded video from a source, decode it,
applying some very simple operations (e.g., adding timestamp) and then write
to a video file, how will these two approaches compare?

* Building an OpenCV instance that supports both method proves no easy task.
Here we document them separately just for the sake of easy debugging.

## Common steps -- prepare Nvidia GPU infrastructure

1. Make sure you have a compatible GPU of course.
    * Nvidia provides a support matrix
  [here](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)

2. Install Nvidia's GPU driver
    [here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#package-manager)

3. Package `nvidia-utils-<version>` should be automatically installed by the
above steps. If so, try `nvidia-smi` to see the status of your GPU.
    * `nvtop` is a usefull third-party tool that demonstrate more GPU details
    in a manner similar to `htop`: `apt install nvtop`.


## OpenCV with `cv::cudacodec`

* Have Nvidia GPU, driver and CUDA library properly installed (verify this
by issuing `nvidia-smi`)

* Download Nvidia's Video Codec SDK. This is a very confusing step as seems
it is documented nowhere except in a stack exchange post
[here](https://stackoverflow.com/questions/65740367/reading-a-video-on-gpu-using-c-and-cuda)
  * Long story short, we need to download Nvidia's Video Codec SDK
  [here](https://developer.nvidia.com/video-codec-sdk) and copy all header files
  in `./Interface/` directory to corresponding CUDA's include directory
  (e.g., `/usr/local/cuda/targets/x86_64-linux/include/`)
  * Without this step, `OpenCV`'s `cmake`/`make` could still work, but the
  compiled code will complain:
    ```
    terminate called after throwing an instance of 'cv::Exception'
    what():  OpenCV(4.7.0) ./repos/opencv/modules/core/include/opencv2/core/private.cuda.hpp:112: error: (-213:The function/feature is not implemented) The called functionality is disabled for current build or platform in function 'throw_no_cuda'
    ```
  * It is also noteworthy that the so called "Nvidia's Video Codec SDK" is the
  same as the "nv-codec-headers.git" mentioned in the [HWAccelIntro](https://trac.ffmpeg.org/wiki/HWAccelIntro)
  of the OpenCV with FFmpeg route documented below. However, FFmpeg states
  that it uses a slightly modified version of Nvidia Video Codec SDK and
  experiment also shows that the headers installed by `nv-codec-headers` won't
  be recognized by OpenCV's `cmake`. So we need two copies of the same set of
  headers files for two routes to work concurrently.

* Prepare [opencv_contrib](https://github.com/opencv/opencv_contrib) repository.
OpenCV needs it to build `cuda` support.

* The final `cmake` command should look like below:

  ```bash
  cmake \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_CUDA=ON \
  -D WITH_NVCUVID=ON \
  -D WITH_NVCUVENC=ON \
  -D BUILD_opencv_cudacodec=ON \
  -D OPENCV_EXTRA_MODULES_PATH=~/repos/opencv_contrib/modules/ \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_PC_FILE_NAME=opencv.pc \
  -D BUILD_EXAMPLES=OFF \
  ..
  ```
  and the `cmake` report should show lines that indicate the inclusion of NVCUVID / NVCUVENC:
  ```
  --   NVIDIA CUDA:                   YES (ver 11.8, CUFFT CUBLAS NVCUVID NVCUVENC)
  --     NVIDIA GPU arch:             35 37 50 52 60 61 70 75 80 86
  --     NVIDIA PTX archs:
  ```

  The same information should be printed when `cv::getBuildInformation()` is called.


## OpenCV with FFmpeg (`avcodec`/`avformat`/etc) that enables CUDA

### I. Build FFmpeg

1. While OpenCV supports multiple backends, it seems that FFmpeg enjoys
better support from Nvidia, making it a good option to work with Nvidia's GPU.

1. If there is an `FFmpeg` installed by `apt`, remove it first.
    * Ubuntu does allow multiple versions to co-exist, the issue is that
    in compilation/linking, it make not be easy to configure different build
    systems (CMake/Ninja/handwritten ones) to use exactly the version of
    FFmpeg we want. Therefore, it is simpler if we can remove other
    unused FFmpeg.
    * Apart from `FFmpeg` itself, it is possible that some of its libraries,
    such as `libswscale` and `libavutil` can exist independently. If so, try
    commands such as `find / -name libswscale.so* 2> /dev/null` to find them
    and then issue commands such as `apt remove libswscale5` to remove them.

1. Build `FFmpeg` with Nvidia GPU support **based on** this
[Nvidia document](https://docs.nvidia.com/video-technologies/video-codec-sdk/pdf/Using_FFmpeg_with_NVIDIA_GPU_Hardware_Acceleration.pdf)
and this [FFmpeg document on HWAccel](https://trac.ffmpeg.org/wiki/HWAccelIntro).
  * We may need to combine options gleaned from different sources to
  construct the final `./configure` command.


1. One working version of `./configure` is: `./configure --enable-pic --enable-shared --enable-nonfree --enable-cuda-nvcc --enable-cuda-llvm --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-cflags="-fPIC" --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags="-gencode arch=compute_52,code=sm_52 -O2"`

    * `--enable-cuda-llvm --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec`: enable Nvidia Video Codec SDK
  ([source](https://trac.ffmpeg.org/wiki/HWAccelIntro))
    * `--enable-pic --enable-shared --extra-cflags="-fPIC"`, used to solve
    the issue during OpenCV build in a later stage: "/usr/bin/ld:
    /usr/local/lib/libavcodec.a(vc1dsp_mmx.o):
    relocation R_X86_64_PC32 against symbol `ff_pw_9' can not be used when
    making a shared object; recompile with -fPIC"
    ([source1](https://www.twblogs.net/a/5ef71a3c209c567d16133dae),
    [source2](https://askubuntu.com/questions/1292968/error-when-installing-opencv-any-version-on-ubuntu-18-04))

1. It it likely that we need to compile FFmpeg multiple times to have the
desired functionalities, to not mess the source directory, it is recommended 
that we build the project "out of tree":
  ```bash
  mkdir /tmp/FFmpeg
  cd /tmp/FFmpeg
  ~/repos/FFmpeg/configure <whatever arguments>    
  ```

1. If `./configure`'s report contains strings `h264_nvenc`/`hevc_nvenc`/etc
like below, the proper hardware encoders/decoders are correctly configured:
  ```
  Enabled encoders:
  ...
  adpcm_argo              aptx                    dvbsub                  h264_nvenc              msmpeg4v2               pcm_s16le_planar        pcm_u8                  roq                     targa                   wmav2
  adpcm_g722              aptx_hd                 dvdsub                  h264_v4l2m2m            msmpeg4v3               pcm_s24be               pcm_vidc                roq_dpcm                text                    wmv1
  adpcm_g726              ass                     dvvideo                 hevc_nvenc              msvideo1                pcm_s24daud             pcx                     rpza                    tiff                    wmv2
  ...
  Enabled hwaccels:
  av1_nvdec               hevc_nvdec              mpeg1_nvdec             mpeg4_nvdec             vp8_nvdec               wmv3_nvdec
  h264_nvdec              mjpeg_nvdec             mpeg2_nvdec             vc1_nvdec               vp9_nvdec

  ```

1. If build is successful, execute `FFmpeg` and check if it works.
    * It should show something like:

    ```
    ffmpeg version n4.4.3-48-gc3ad886251 Copyright (c) 2000-2022 the FFmpeg developers
    built with gcc 11 (Ubuntu 11.3.0-1ubuntu1~22.04)
    configuration: --enable-pic --enable-shared --enable-nonfree --enable-cuda-sdk --enable-cuda-llvm --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-cflags=-fPIC --extra-ldflags=-L/usr/local/cuda/lib64
    libavutil      56. 70.100 / 56. 70.100
    libavcodec     58.134.100 / 58.134.100
    ...
    ```
    * If Nvidia's GPU acceleration is compiled in, issuing
    `ffmpeg -codecs | grep h264` should show something like:
    ```
    ...
    DEV.LS h264                 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (decoders: h264 h264_v4l2m2m h264_cuvid ) (encoders: h264_nvenc h264_v4l2m2m nvenc nvenc_h264 )
    ```

### II. Build OpenCV

1. The first perennial problem when building OpenCV on top of a custom FFmpeg
build is compatibility--each OpenCV version depends on one or few snapshots of
FFmpeg versions, if the versions of OpenCV and FFmpeg mismatch, various
compilation errors would occur.
    * After multiple attempts, this note is prepared with FFmpeg `n4.4.3`
    and OpenCV `4.6.0`.

1. Issue `cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX="$HOME/.local"  ..`
to install the built OpenCV to the user's home directory.
    * Verify the result of `cmake` carefully before proceeding to `make`,
    make sure that `cmake` shows the expected version of `FFmpeg`.

1. If OpenCV fails to compile due to a compatibility issue with FFmpeg, we need
to go back to `I` and build FFmpeg again.
    * Before issuing `git checkout <branch/tag>`, one needs to issue
    `sudo make uninstall` and `sudo make distclean` to make sure the previous
    FFmpeg build is cleared; otherwise multiple FFmpeg versions could co-exist,
    causing confusion and strange errors.

## III. Set environment variables

* TL;DR: add the below to `~/.profile`/`~/.bashrc`
```
export OPENCV_VIDEOIO_DEBUG=0
export OPENCV_FFMPEG_DEBUG=0
export OPENCV_LOG_LEVEL=DEBUG
export OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;cuvid|video_codec;h264_cuvid"
#export OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;cuvid|video_codec;mjpeg_cuvid"
export OPENCV_FFMPEG_WRITER_OPTIONS="hw_encoders_any;cuda"
```

1. Enable OpenCV's debug logging, which could reveal useful information:

    ```bash
    export OPENCV_VIDEOIO_DEBUG=1
    export OPENCV_FFMPEG_DEBUG=1
    export OPENCV_LOG_LEVEL=DEBUG
    ```

1. Enable hardware decoding for `cv::VideoCapture` with environment variable
`OPENCV_FFMPEG_CAPTURE_OPTIONS`. This envvar uses a special key/value pair
format `key1;val1|key2;val2`. To use hardware decoder, we want to set it
to something like:
  * `"hwaccel;cuvid|video_codec;mjpeg_cuvid"`--if we know the video feed is in
  MJPG format.
  * `"hwaccel;cuvid|video_codec;h264_cuvid"`--if we know the video feed is in
  h264 format.
  * List the supported codec by `ffmpeg -codecs | grep nv`.

1. Enable hardware decoding for `cv::VideoWriter` with environment variable
`OPENCV_FFMPEG_WRITER_OPTIONS`. This envvar uses a special key/value pair
format `key1;val1|key2;val2`. To use hardware encoder, we want to set it
to `"hw_encoders_any;cuda"`
    * It appears that OpenCV does not allow us to pick a specific encoder.
