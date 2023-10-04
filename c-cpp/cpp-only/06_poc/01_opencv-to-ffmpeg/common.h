#include <string>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

string ffmpegCommand = "/usr/bin/ffmpeg -y -f rawvideo -pixel_format bgr24 "
    "-video_size 1920x1080 -framerate 30 -i pipe:0 /tmp/test.mp4";

VideoCapture openVideoSource() {
    VideoCapture cap = VideoCapture();
    /* Hint: OPENCV_TEST_URI_PATH should be something like:
        * /dev/video0
        * rtsp://localhost:8554/test
        * /tmp/sample-video.mp4 */
    cout << "OPENCV_TEST_URI_PATH: " << getenv("OPENCV_TEST_URI_PATH") << endl;
    if (cap.open(getenv("OPENCV_TEST_URI_PATH")) == false) {
        cerr << "Failed to open, exiting..." << endl;
        abort();
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // Hopefully we have copy elision/RVO here.
    return cap;
}
