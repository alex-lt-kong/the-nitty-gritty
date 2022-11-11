# Dependencies

* OpenCV
* ffmpeg

# Compilation
```
g++ main.cpp -o main -L/usr/local/lib -I/usr/local/include/opencv4 -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_core
```

# `main0` vs `main1`
* `main1` is a better approach as it achieves full 30 fps while `main0` can only achieve around 15 fps.
* Why? `main1` has far fewer `fwrite()` call which saves a lot of time.