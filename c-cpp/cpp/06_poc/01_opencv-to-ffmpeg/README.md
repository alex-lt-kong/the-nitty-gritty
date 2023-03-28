# Piping OpenCV frames to FFmpeg

## Dependencies

* OpenCV: `apt install libopencv-dev`
* FFmpeg: `apt install ffmpeg`

## Comparison

* On the same machine without using GPU, `2_popen-batched.cpp` is roughly 2x
to 5x as fast as `1_popen-naive.cpp`.
    * `1_popen-naive.cpp` won't be able to saturate even one CPU core.