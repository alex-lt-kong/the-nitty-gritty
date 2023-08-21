
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "utils.h"

int main(int argc, const char *argv[]) {
  std::cout << cv::getBuildInformation() << std::endl;
  if (argc != 3) {
    std::cerr << "Usage : " << argv[0] << "  <Source URI> <Dest path>"
              << std::endl;
    return -1;
  }

  install_signal_handler();
  std::cout << "A signal handler is installed, "
               "press Ctrl+C to exit the event loop gracefully."
            << std::endl;

  cv::VideoCapture cap;
  bool result = cap.open(std::string(argv[1]), cv::CAP_FFMPEG,
  {cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});
  if (!result) {
    std::cerr << "Error!" << std::endl;
  }
  cv::VideoWriter vwriter = cv::VideoWriter(
      std::string(argv[2]), cv::CAP_ANY,
      cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
      25.0, cv::Size(1280, 720)  ,
      {cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});

  cv::Mat hFrame;
  size_t frameCount = 0;
  while (!e_flag) {

    if (!cap.read(hFrame)) {
      std::cerr << "cap.read(hFrame) is False" << std::endl;
      break;
    }

    std::cout << "frameCount: " << ++frameCount << ", size(): " << hFrame.size()
              << ", channels(): " << hFrame.channels() << std::endl;
    if (!hFrame.empty()) {
      vwriter.write(hFrame);
    }
  }
  vwriter.release();
  std::cout << "vwriter.release()ed" << std::endl;
  return 0;
}
