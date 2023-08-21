
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

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

  cv::cuda::GpuMat dFrameCurr, dFramePrev;

  cv::Ptr<cv::cudacodec::VideoReader> dReader =
      cv::cudacodec::createVideoReader(string(argv[1]));
  cv::Ptr<cv::cudacodec::VideoWriter> dWriter =
      cv::cudacodec::createVideoWriter(string(argv[2]), cv::Size(1280, 720),
                                       cv::cudacodec::Codec::H264, 25.0,
                                       cv::cudacodec::ColorFormat::BGRA);
  size_t frameCount = 0;
  while (!e_flag) {
    if (!dFrameCurr.empty()) {
      dFramePrev = dFrameCurr.clone();
    }
    if (!dReader->nextFrame(dFrameCurr)) {
      cerr << "dReader->nextFrame(dFrame) is False" << endl;
      break;
    }
    ++frameCount;
    auto start = chrono::high_resolution_clock::now();
    float diff = getCudaFrameChanges(dFramePrev, dFrameCurr);
    auto end = chrono::high_resolution_clock::now();
    if (!dFramePrev.empty() && frameCount % 10 == 0) {
      cout << "frameCount: " << frameCount << ", size(): " << dFrameCurr.size()
           << ", channels(): " << dFrameCurr.channels() << ", diff: " << diff
           << "("
           << chrono::duration_cast<chrono::milliseconds>(end - start).count()
           << " ms)" << endl;
    }
    if (!dFrameCurr.empty()) {
      dWriter->write(dFrameCurr);
    }
  }
  dWriter->release();
  cout << "dWriter->release()ed" << endl;
  return 0;
}
