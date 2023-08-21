
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

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

  cv::cuda::GpuMat dFrame;

  cv::Ptr<cv::cudacodec::VideoReader> dReader =
      cv::cudacodec::createVideoReader(std::string(argv[1]));
  cv::Ptr<cv::cudacodec::VideoWriter> dWriter =
      cv::cudacodec::createVideoWriter(
          std::string(argv[2]), cv::Size(1280, 720), cv::cudacodec::Codec::H264,
          25.0, cv::cudacodec::ColorFormat::BGRA);
  size_t frameCount = 0;
  while (!e_flag) {

    if (!dReader->nextFrame(dFrame)) {
      std::cerr << "dReader->nextFrame(dFrame) is False" << std::endl;
      break;
    }

    std::cout << "frameCount: " << ++frameCount << ", size(): " << dFrame.size()
              << ", channels(): " << dFrame.channels() << std::endl;
    if (!dFrame.empty()) {
      dWriter->write(dFrame);
    }
  }
  dWriter->release();
  std::cout << "dWriter->release()ed" << std::endl;
  return 0;
}
