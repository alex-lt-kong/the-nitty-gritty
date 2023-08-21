
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "utils.hpp"

using namespace std;

int main(int argc, const char *argv[]) {
  cout << cv::getBuildInformation() << endl;
  if (argc != 3) {
    cerr << "Usage : " << argv[0] << "  <Source URI> <Dest path>" << endl;
    return -1;
  }

  install_signal_handler();
  cout << "A signal handler is installed, "
          "press Ctrl+C to exit the event loop gracefully."
       << endl;

  cv::VideoCapture cap;
  bool result =
      cap.open(string(argv[1]), cv::CAP_ANY,
               {cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});
  if (!result) {
    cerr << "Error!" << endl;
  }
  cv::VideoWriter vwriter = cv::VideoWriter(
      string(argv[2]), cv::CAP_ANY, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
      25.0, cv::Size(1280, 720),
      {cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});

  cv::Mat hFrameCurr, hFramePrev;
  size_t frameCount = 0;
  while (!e_flag) {

    if (!hFrameCurr.empty()) {
      hFramePrev = hFrameCurr.clone();
    }
    if (!cap.read(hFrameCurr)) {
      cerr << "cap.read(hFrame) is False" << endl;
      break;
    }
    ++frameCount;
    auto start = chrono::high_resolution_clock::now();
    float diff = getFrameChanges(hFramePrev, hFrameCurr);
    auto end = chrono::high_resolution_clock::now();
    if (!hFramePrev.empty() && frameCount % 10 == 0) {
      cout << "frameCount: " << frameCount << ", size(): " << hFrameCurr.size()
           << ", channels(): " << hFrameCurr.channels() << ", diff: " << diff
           << "("
           << chrono::duration_cast<chrono::milliseconds>(end - start).count()
           << " ms)" << endl;
    }
    if (!hFrameCurr.empty()) {
      vwriter.write(hFrameCurr);
    }
  }
  vwriter.release();
  cout << "vwriter.release()ed" << endl;
  return 0;
}
