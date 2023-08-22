
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char *argv[]) {
  cout << getBuildInformation() << endl;
  if (argc != 3) {
    cerr << "Usage : " << argv[0] << "  <Source URI> <Dest path>" << endl;
    return -1;
  }

  install_signal_handler();
  cout << "A signal handler is installed, "
          "press Ctrl+C to exit the event loop gracefully."
       << endl;

  cuda::GpuMat dFrameCurr, dFramePrev;
  Mat hFrame;
  Ptr<cudacodec::VideoReader> dReader =
      cudacodec::createVideoReader(string(argv[1]));

  dReader->set(cv::cudacodec::ColorFormat::BGR);
  VideoWriter hWriter = VideoWriter(
      string(argv[2]), CAP_ANY, VideoWriter::fourcc('m', 'p', '4', 'v'), 25.0,
      Size(1280, 720),
      {VIDEOWRITER_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY});
  size_t frameCount = 0;
  while (!e_flag) {
    auto t0 = chrono::high_resolution_clock::now();

    if (!dReader->nextFrame(dFrameCurr)) {
      cerr << "dReader->nextFrame(dFrame) is False" << endl;
      break;
    }
    ++frameCount;
    auto t1 = chrono::high_resolution_clock::now();
    float diff = getCudaFrameChanges(dFramePrev, dFrameCurr);
    auto t2 = chrono::high_resolution_clock::now();

    if (!dFrameCurr.empty()) {
      dFrameCurr.download(hFrame);
      dFramePrev = dFrameCurr.clone();
      hWriter.write(hFrame);
    }
    auto t3 = chrono::high_resolution_clock::now();
    if (!dFramePrev.empty() && frameCount % 10 == 0) {
      cout << "frameCount: " << frameCount << ", size(): " << dFrameCurr.size()
           << ", channels(): " << dFrameCurr.channels() << ", diff: " << diff
           << "("
           << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()
           << " ms) , iteration took: "
           << chrono::duration_cast<chrono::milliseconds>(t3 - t1).count()
           << " ms\n";
    }
  }
  hWriter.release();
  cout << "hWriter.release()ed" << endl;
  return 0;
}
