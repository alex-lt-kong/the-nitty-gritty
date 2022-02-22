# Dependencies

* OpenCV
* ffmpeg

# Compilation
```
g++ main.cpp -o main -L/usr/local/lib -I/usr/local/include/opencv4 -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_core
```

# Sample output
```
> ./main 
ffmpeg version 4.3.3-0+deb11u1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 10 (Debian 10.2.1-6)
  configuration: --prefix=/usr --extra-version=0+deb11u1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 51.100 / 56. 51.100
  libavcodec     58. 91.100 / 58. 91.100
  libavformat    58. 45.100 / 58. 45.100
  libavdevice    58. 10.100 / 58. 10.100
  libavfilter     7. 85.100 /  7. 85.100
  libavresample   4.  0.  0 /  4.  0.  0
  libswscale      5.  7.100 /  5.  7.100
  libswresample   3.  7.100 /  3.  7.100
  libpostproc    55.  7.100 / 55.  7.100
Input #0, rawvideo, from 'pipe:0':
  Duration: N/A, start: 0.000000, bitrate: 497664 kb/s
    Stream #0:0: Video: rawvideo (BGR[24] / 0x18524742), bgr24, 1920x1080, 497664 kb/s, 10 tbr, 10 tbn, 10 tbc
Stream mapping:
  Stream #0:0 -> #0:0 (rawvideo (native) -> h264 (libx264))
[libx264 @ 0x561eb3c21500] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2
[libx264 @ 0x561eb3c21500] profile High 4:4:4 Predictive, level 4.0, 4:4:4, 8-bit
[libx264 @ 0x561eb3c21500] 264 - core 160 r3011 cde9a93 - H.264/MPEG-4 AVC codec - Copyleft 2003-2020 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=4 threads=12 lookahead_threads=2 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=10 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mp4, to './test.mp4':
  Metadata:
    encoder         : Lavf58.45.100
    Stream #0:0: Video: h264 (libx264) (avc1 / 0x31637661), yuv444p, 1920x1080, q=-1--1, 10 fps, 10240 tbn, 10 tbc
    Metadata:
      encoder         : Lavc58.91.100 libx264
    Side data:
      cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
frame=  101 fps=6.7 q=-1.0 Lsize=    3091kB time=00:00:09.80 bitrate=2583.4kbits/s speed=0.649x    
video:3088kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.066211%
[libx264 @ 0x561eb3c21500] frame I:1     Avg QP:15.43  size:298115
[libx264 @ 0x561eb3c21500] frame P:28    Avg QP:16.80  size: 84734
[libx264 @ 0x561eb3c21500] frame B:72    Avg QP:22.79  size:  6823
[libx264 @ 0x561eb3c21500] consecutive B-frames:  4.0%  2.0%  3.0% 91.1%
[libx264 @ 0x561eb3c21500] mb I  I16..4:  7.6% 60.4% 32.0%
[libx264 @ 0x561eb3c21500] mb P  I16..4:  0.6%  1.6%  0.4%  P16..4: 36.2% 13.6% 12.5%  0.0%  0.0%    skip:35.1%
[libx264 @ 0x561eb3c21500] mb B  I16..4:  0.0%  0.0%  0.0%  B16..8: 29.4%  2.2%  1.3%  direct: 1.0%  skip:66.1%  L0:36.9% L1:57.9% BI: 5.2%
[libx264 @ 0x561eb3c21500] 8x8 transform intra:60.5% inter:71.3%
[libx264 @ 0x561eb3c21500] coded y,u,v intra: 70.2% 23.8% 17.3% inter: 12.2% 1.7% 1.5%
[libx264 @ 0x561eb3c21500] i16 v,h,dc,p:  5% 53% 11% 31%
[libx264 @ 0x561eb3c21500] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 10% 39% 21%  4%  4%  3%  7%  3%  8%
[libx264 @ 0x561eb3c21500] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 17% 33% 15%  5%  7%  5%  8%  4%  7%
[libx264 @ 0x561eb3c21500] Weighted P-Frames: Y:0.0% UV:0.0%
[libx264 @ 0x561eb3c21500] ref P L0: 68.7%  6.4% 18.6%  6.4%
[libx264 @ 0x561eb3c21500] ref B L0: 89.1%  9.5%  1.3%
[libx264 @ 0x561eb3c21500] ref B L1: 94.3%  5.7%
[libx264 @ 0x561eb3c21500] kb/s:2504.49

```