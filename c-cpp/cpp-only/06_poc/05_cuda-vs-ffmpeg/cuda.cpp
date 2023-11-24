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
  cv::cuda::GpuMat diffFrame;
  Mat hFrame;

  Ptr<cudacodec::VideoReader> dReader =
      cudacodec::createVideoReader(string(argv[1]));
  dReader->set(cv::cudacodec::ColorFormat::BGR);
  Ptr<cudacodec::VideoWriter> dWriter = cudacodec::createVideoWriter(
      string(argv[2]), Size(1280, 720), cudacodec::Codec::H264, 25.0,
      cudacodec::ColorFormat::BGR);
  size_t frameCount = 0;
  while (!e_flag) {
    if (!dReader->nextFrame(dFrameCurr)) {
      cerr << "dReader->nextFrame(dFrame) is False" << endl;
      this_thread::sleep_for(10000ms);
      dReader = cudacodec::createVideoReader(string(argv[1]));
    }
    ++frameCount;
    auto t1 = chrono::high_resolution_clock::now();
    float diff = getCudaFrameChanges(dFramePrev, dFrameCurr, diffFrame);
    auto t2 = chrono::high_resolution_clock::now();

    if (!dFrameCurr.empty()) {
      cuda::GpuMat t = dFrameCurr.clone();
      // cuda::rotate(dFrameCurr, dFrameCurr, dFrameCurr.size(), 180);
      dFramePrev = dFrameCurr.clone();
      dFrameCurr.download(hFrame);
      overlayDatetime(hFrame);
      // Need to emulate this download()/upload() cycle
      dFrameCurr.upload(hFrame);
      dWriter->write(dFrameCurr);
    } else {
      cerr << "frameCount: " << frameCount << " is empty" << endl;
    }
    auto t3 = chrono::high_resolution_clock::now();
    if (!dFramePrev.empty() && frameCount % 100 == 0) {
      cout << "frameCount: " << frameCount << ", size(): " << dFrameCurr.size()
           << ", channels(): " << dFrameCurr.channels() << ", diff: " << fixed
           << setprecision(2) << diff << "% ("
           << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()
           << " ms) , iteration took: "
           << chrono::duration_cast<chrono::milliseconds>(t3 - t1).count()
           << " ms" << endl;
    }
  }
  dWriter->release();
  cout << "dWriter->release()ed" << endl;
  return 0;
}
